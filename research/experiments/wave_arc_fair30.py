#!/usr/bin/env python3
"""Fair 30-task ARC comparison: complex wave model vs parameter-matched real recurrent control.

The task sample is exactly the deterministic sample used by wave_arc_kill_test.py:
selection seed 20260915, shape-preserving tasks, max 400 cells.
No task-specific tuning is performed.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_arc_kill_test import (
    N_COLORS,
    NeighborWaveARC,
    cell_color_map_score,
    choose_tasks,
    count_parameters,
    load_task,
    train_model,
)


class RealMatchedARC(nn.Module):
    """Near-parameter-matched real-valued control.

    It keeps the wave model's key non-semantic constraints:
      * fixed local neighbor propagation;
      * trainable local mode mixing shared at every grid site;
      * recurrent processing with the same number of steps;
      * persistent input injection.

    It removes complex phase/interference: state and couplings are real and the
    nonlinearity is a conventional saturating tanh.  With 13 channels it has
    442 trainable real parameters vs 456 for the default 8-complex-channel
    wave model.
    """

    def __init__(self, channels: int = 13, steps: int = 4, seed: int = 0):
        super().__init__()
        self.channels = channels
        self.steps = steps
        g = torch.Generator().manual_seed(seed)
        scale = 0.20 / math.sqrt(channels)

        self.embed = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.mix = nn.Parameter(torch.eye(channels) + 0.04 * torch.randn(channels, channels, generator=g))
        self.out = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.gain = nn.Parameter(0.03 * torch.randn(channels, generator=g))

        amp = torch.tensor([0.55, 0.16, 0.16, 0.16, 0.16])
        sign = torch.where(torch.rand(channels, 5, generator=g) > 0.5, 1.0, -1.0)
        self.register_buffer("coupling", amp[None, :] * sign)

    @staticmethod
    def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        padded = F.pad(x, (1, 1, 1, 1))
        h, w = x.shape[-2:]
        return padded[..., 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    def _propagate(self, state: torch.Tensor) -> torch.Tensor:
        neighbors = [
            state,
            self._shift(state, 1, 0),
            self._shift(state, -1, 0),
            self._shift(state, 0, 1),
            self._shift(state, 0, -1),
        ]
        out = torch.zeros_like(state)
        for k in range(5):
            out = out + neighbors[k] * self.coupling[:, k][None, :, None, None]
        return out

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = grid[None]
        excitation = self.embed[x].permute(0, 3, 1, 2)
        state = excitation

        for _ in range(self.steps):
            propagated = self._propagate(state)
            mixed = torch.einsum("bchw,dc->bdhw", propagated, self.mix)
            gain = 1.0 + self.gain[None, :, None, None]
            state = torch.tanh(mixed * gain) + 0.25 * excitation

        return torch.einsum("bchw,kc->bkhw", state, self.out)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--selection-seed", type=int, default=20260915)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--wave-channels", type=int, default=8)
    parser.add_argument("--real-channels", type=int, default=13)
    parser.add_argument("--output", type=Path, default=Path("research/experiments/wave_arc_fair30_results.csv"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    seeds = parse_seeds(args.seeds)
    task_paths = choose_tasks(args.data_dir, args.limit, args.selection_seed, args.max_cells, None)

    probe_wave = NeighborWaveARC(args.wave_channels, args.steps, 0)
    probe_real = RealMatchedARC(args.real_channels, args.steps, 0)
    print("wave_parameters", count_parameters(probe_wave), flush=True)
    print("real_parameters", count_parameters(probe_real), flush=True)
    print("task_ids", ",".join(p.stem for p in task_paths), flush=True)

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

        for seed in seeds:
            for name, model in (
                ("wave", NeighborWaveARC(args.wave_channels, args.steps, seed)),
                ("real_matched", RealMatchedARC(args.real_channels, args.steps, seed)),
            ):
                train_score, test_score, elapsed = train_model(model, task, args.iterations, args.lr)
                row = {
                    "task_id": task_id,
                    "model": name,
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
    fields = [
        "task_id", "model", "seed", "parameters", "train_exact", "train_pixel",
        "test_exact", "test_pixel", "seconds",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSUMMARY", flush=True)
    for model_name in ("wave", "real_matched"):
        rr = [r for r in rows if r["model"] == model_name]
        exact_runs = sum(float(r["test_exact"]) == 1.0 for r in rr)
        robust_tasks = 0
        majority_tasks = 0
        for task_id in [p.stem for p in task_paths]:
            tr = [r for r in rr if r["task_id"] == task_id]
            wins = sum(float(r["test_exact"]) == 1.0 for r in tr)
            robust_tasks += wins == len(seeds)
            majority_tasks += wins >= (len(seeds) // 2 + 1)
        mean_pixel = sum(float(r["test_pixel"]) for r in rr) / len(rr)
        mean_train_exact = sum(float(r["train_exact"]) for r in rr) / len(rr)
        print(
            model_name,
            "exact_task_seed_runs", f"{exact_runs}/{len(rr)}",
            "robust_tasks", f"{robust_tasks}/{len(task_paths)}",
            "majority_tasks", f"{majority_tasks}/{len(task_paths)}",
            "mean_test_pixel", round(mean_pixel, 6),
            "mean_train_exact", round(mean_train_exact, 6),
            flush=True,
        )

    print("results", args.output, flush=True)


if __name__ == "__main__":
    main()
