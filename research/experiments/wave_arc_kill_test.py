#!/usr/bin/env python3
"""Zero-budget ARC-AGI kill test for a wave-inspired recurrent grid learner.

This is deliberately not a general ARC solver. It restricts the first gate to
shape-preserving tasks so output-size inference does not confound the question:
can a shared-parameter physical/wave-inspired learner infer reusable spatial
transformations from a few demonstrations?

Usage from repository root:
    python research/experiments/wave_arc_kill_test.py --limit 30 --seeds 0,1,2
    python research/experiments/wave_arc_kill_test.py --task-ids a699fb00,a85d4709

Outputs a CSV with exact-match and pixel accuracy for each model/seed/task.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


N_COLORS = 10


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def shape_preserving(task: dict) -> bool:
    examples = list(task.get("train", [])) + list(task.get("test", []))
    if not examples:
        return False
    for ex in examples:
        if grid_shape(ex["input"]) != grid_shape(ex["output"]):
            return False
    return True


def max_grid_cells(task: dict) -> int:
    examples = list(task.get("train", [])) + list(task.get("test", []))
    return max(grid_shape(ex["input"])[0] * grid_shape(ex["input"])[1] for ex in examples)


def load_task(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NeighborWaveARC(nn.Module):
    """Small recurrent wave-inspired cellular learner.

    Physics-inspired constraints:
      * fixed complex spatial propagation coefficients;
      * trainable local mode mixing shared at every grid site;
      * intensity-dependent phase response and amplitude saturation;
      * square-law detector readout.

    It is intentionally tiny and spatially shared. It cannot allocate a free
    parameter to every ARC pixel, which would make the few-shot experiment
    meaningless.
    """

    def __init__(self, channels: int = 8, steps: int = 4, seed: int = 0):
        super().__init__()
        self.channels = channels
        self.steps = steps
        g = torch.Generator().manual_seed(seed)
        scale = 0.20 / math.sqrt(channels)

        self.embed_r = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.embed_i = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.mix_r = nn.Parameter(torch.eye(channels) + 0.04 * torch.randn(channels, channels, generator=g))
        self.mix_i = nn.Parameter(0.04 * torch.randn(channels, channels, generator=g))
        self.out_r = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.out_i = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.phase_beta = nn.Parameter(0.03 * torch.randn(channels, generator=g))

        amp = torch.tensor([0.55, 0.16, 0.16, 0.16, 0.16])
        phase = 2 * math.pi * torch.rand(channels, 5, generator=g)
        coeff = amp[None, :] * torch.exp(1j * phase)
        self.register_buffer("coupling_r", coeff.real)
        self.register_buffer("coupling_i", coeff.imag)

    @staticmethod
    def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        padded = F.pad(x, (1, 1, 1, 1))
        h, w = x.shape[-2:]
        return padded[..., 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    def _propagate(self, hr: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_neighbors = [
            hr,
            self._shift(hr, 1, 0),
            self._shift(hr, -1, 0),
            self._shift(hr, 0, 1),
            self._shift(hr, 0, -1),
        ]
        imag_neighbors = [
            hi,
            self._shift(hi, 1, 0),
            self._shift(hi, -1, 0),
            self._shift(hi, 0, 1),
            self._shift(hi, 0, -1),
        ]
        rr = torch.zeros_like(hr)
        ii = torch.zeros_like(hi)
        for k in range(5):
            cr = self.coupling_r[:, k][None, :, None, None]
            ci = self.coupling_i[:, k][None, :, None, None]
            rr = rr + real_neighbors[k] * cr - imag_neighbors[k] * ci
            ii = ii + real_neighbors[k] * ci + imag_neighbors[k] * cr
        return rr, ii

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = grid[None]
        er = self.embed_r[x].permute(0, 3, 1, 2)
        ei = self.embed_i[x].permute(0, 3, 1, 2)
        hr, hi = er, ei

        for _ in range(self.steps):
            pr, pi = self._propagate(hr, hi)
            rr = torch.einsum("bchw,dc->bdhw", pr, self.mix_r) - torch.einsum("bchw,dc->bdhw", pi, self.mix_i)
            ri = torch.einsum("bchw,dc->bdhw", pr, self.mix_i) + torch.einsum("bchw,dc->bdhw", pi, self.mix_r)
            amp2 = rr.square() + ri.square()
            phase = self.phase_beta[None, :, None, None] * amp2
            saturation = torch.rsqrt(1.0 + 0.15 * amp2)
            hr = saturation * (rr * torch.cos(phase) - ri * torch.sin(phase)) + 0.25 * er
            hi = saturation * (rr * torch.sin(phase) + ri * torch.cos(phase)) + 0.25 * ei

        fr = torch.einsum("bchw,kc->bkhw", hr, self.out_r) - torch.einsum("bchw,kc->bkhw", hi, self.out_i)
        fi = torch.einsum("bchw,kc->bkhw", hr, self.out_i) + torch.einsum("bchw,kc->bkhw", hi, self.out_r)
        power = fr.square() + fi.square() + 1e-7
        return torch.log(power)


class RecurrentConvARC(nn.Module):
    """Small conventional recurrent CNN baseline with coordinate channels."""

    def __init__(self, channels: int = 8, steps: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.steps = steps
        self.input_conv = nn.Conv2d(N_COLORS + 2, channels, 3, padding=1)
        self.recurrent = nn.Conv2d(channels, channels, 3, padding=1)
        self.output_conv = nn.Conv2d(channels, N_COLORS, 1)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = grid[None]
        _, h, w = x.shape
        one_hot = F.one_hot(x, N_COLORS).permute(0, 3, 1, 2).float()
        yy = torch.linspace(-1.0, 1.0, h, device=x.device)[None, None, :, None].expand(1, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device)[None, None, None, :].expand(1, 1, h, w)
        inp = torch.cat([one_hot, yy, xx], dim=1)
        state = torch.relu(self.input_conv(inp))
        for _ in range(self.steps):
            state = torch.relu(self.recurrent(state) + state)
        return self.output_conv(state)


@dataclass
class Score:
    exact_fraction: float
    pixel_accuracy: float


def score_examples(model: nn.Module, examples: list[dict]) -> Score:
    exact = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in examples:
            x = torch.tensor(ex["input"], dtype=torch.long)
            y = torch.tensor(ex["output"], dtype=torch.long)
            pred = model(x).argmax(1)[0]
            exact += int(torch.equal(pred, y))
            correct += int((pred == y).sum())
            total += y.numel()
    return Score(exact / len(examples), correct / total)


def class_weights(examples: list[dict]) -> torch.Tensor:
    counts = torch.ones(N_COLORS)
    for ex in examples:
        y = torch.tensor(ex["output"], dtype=torch.long)
        counts += torch.bincount(y.flatten(), minlength=N_COLORS).float()
    weights = torch.sqrt(counts.sum() / counts)
    return weights / weights.mean()


def train_model(model: nn.Module, task: dict, iterations: int, lr: float) -> tuple[Score, Score, float]:
    examples = [
        (torch.tensor(ex["input"], dtype=torch.long), torch.tensor(ex["output"], dtype=torch.long))
        for ex in task["train"]
    ]
    weights = class_weights(task["train"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.tensor(0.0)
        for x, y in examples:
            loss = loss + F.cross_entropy(model(x), y[None], weight=weights)
        loss = loss / len(examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    elapsed = time.perf_counter() - start
    return score_examples(model, task["train"]), score_examples(model, task["test"]), elapsed


def cell_color_map_score(task: dict) -> Score:
    counts = torch.zeros(N_COLORS, N_COLORS, dtype=torch.long)
    for ex in task["train"]:
        x = torch.tensor(ex["input"], dtype=torch.long)
        y = torch.tensor(ex["output"], dtype=torch.long)
        for a, b in zip(x.flatten(), y.flatten()):
            counts[a, b] += 1
    mapping = counts.argmax(1)
    exact = 0
    correct = 0
    total = 0
    for ex in task["test"]:
        x = torch.tensor(ex["input"], dtype=torch.long)
        y = torch.tensor(ex["output"], dtype=torch.long)
        pred = mapping[x]
        exact += int(torch.equal(pred, y))
        correct += int((pred == y).sum())
        total += y.numel()
    return Score(exact / len(task["test"]), correct / total)


def choose_tasks(data_dir: Path, limit: int, selection_seed: int, max_cells: int, task_ids: list[str] | None) -> list[Path]:
    if task_ids:
        paths = [data_dir / f"{task_id}.json" for task_id in task_ids]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing task files: " + ", ".join(missing))
        return paths

    eligible = []
    for path in sorted(data_dir.glob("*.json")):
        task = load_task(path)
        if shape_preserving(task) and max_grid_cells(task) <= max_cells:
            eligible.append(path)
    rng = random.Random(selection_seed)
    rng.shuffle(eligible)
    return eligible[:limit]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--task-ids", type=str, default="")
    parser.add_argument("--selection-seed", type=int, default=20260915)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--models", type=str, default="wave,conv")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("wave_arc_kill_test_results.csv"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    task_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()] or None
    seeds = parse_csv_ints(args.seeds)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    task_paths = choose_tasks(args.data_dir, args.limit, args.selection_seed, args.max_cells, task_ids)

    rows: list[dict] = []
    for path in task_paths:
        task = load_task(path)
        task_id = path.stem
        trivial = cell_color_map_score(task)
        rows.append({
            "task_id": task_id,
            "model": "cell_color_map",
            "seed": "",
            "parameters": 0,
            "train_exact": "",
            "train_pixel": "",
            "test_exact": trivial.exact_fraction,
            "test_pixel": trivial.pixel_accuracy,
            "seconds": 0.0,
        })

        for model_name in models:
            for seed in seeds:
                if model_name == "wave":
                    model = NeighborWaveARC(args.channels, args.steps, seed)
                elif model_name == "conv":
                    model = RecurrentConvARC(args.channels, args.steps, seed)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                train_score, test_score, elapsed = train_model(model, task, args.iterations, args.lr)
                row = {
                    "task_id": task_id,
                    "model": model_name,
                    "seed": seed,
                    "parameters": count_parameters(model),
                    "train_exact": train_score.exact_fraction,
                    "train_pixel": train_score.pixel_accuracy,
                    "test_exact": test_score.exact_fraction,
                    "test_pixel": test_score.pixel_accuracy,
                    "seconds": elapsed,
                }
                rows.append(row)
                print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id", "model", "seed", "parameters", "train_exact", "train_pixel",
        "test_exact", "test_pixel", "seconds",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nExact successes by model:")
    for name in ["cell_color_map"] + models:
        rr = [r for r in rows if r["model"] == name]
        if name == "cell_color_map":
            solved = sum(float(r["test_exact"]) == 1.0 for r in rr)
            print(f"  {name}: {solved}/{len(rr)} tasks")
        else:
            solved_pairs = sum(float(r["test_exact"]) == 1.0 for r in rr)
            print(f"  {name}: {solved_pairs}/{len(rr)} task-seed runs")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
