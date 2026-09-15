#!/usr/bin/env python3
"""Final bounded continuation gate for the wave-ARC hypothesis.

Compares the simplest surviving complex model (complex-linear recurrent field +
square-law detector) against a near-parameter-matched *real* linear recurrent
model with the same rank-2 quadratic/square-law detector idea.

Two deterministic task slices are supported:
  offset=0   : the original frozen 30 tasks used during development.
  offset=30  : the next 30 tasks from the same deterministic shuffled eligible
               pool; these are untouched by architecture selection.

No task-specific tuning is performed.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_arc_ablation import AblatedWaveARC
from wave_arc_kill_test import (
    N_COLORS,
    count_parameters,
    load_task,
    max_grid_cells,
    shape_preserving,
    train_model,
)


class RealQuadraticARC(nn.Module):
    """Near-parameter-matched real control with quadratic readout.

    Keeps:
      * fixed local neighbor propagation;
      * trainable local shared mode mixing;
      * recurrent linear state updates with persistent input injection;
      * two detector quadratures per class and square-law readout.

    Removes:
      * complex-valued state;
      * phase rotations / complex interference constraints.

    At 11 channels this has 451 trainable real parameters, close to the
    456-parameter full wave model and 448 effective trainable parameters in the
    linear-wave ablation (the unused phase_beta remains allocated there).
    """

    def __init__(self, channels: int = 11, steps: int = 4, seed: int = 0):
        super().__init__()
        self.channels = channels
        self.steps = steps
        g = torch.Generator().manual_seed(seed)
        scale = 0.20 / math.sqrt(channels)

        self.embed = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.mix = nn.Parameter(torch.eye(channels) + 0.04 * torch.randn(channels, channels, generator=g))
        # Two real detector quadratures.  Their squared sum gives the same
        # detector nonlinearity order as |complex amplitude|^2.
        self.out_a = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))
        self.out_b = nn.Parameter(scale * torch.randn(N_COLORS, channels, generator=g))

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
            state = torch.einsum("bchw,dc->bdhw", propagated, self.mix) + 0.25 * excitation

        qa = torch.einsum("bchw,kc->bkhw", state, self.out_a)
        qb = torch.einsum("bchw,kc->bkhw", state, self.out_b)
        power = qa.square() + qb.square() + 1e-7
        return torch.log(power)


def choose_task_slice(
    data_dir: Path,
    limit: int,
    offset: int,
    selection_seed: int,
    max_cells: int,
) -> list[Path]:
    eligible: list[Path] = []
    for path in sorted(data_dir.glob("*.json")):
        task = load_task(path)
        if shape_preserving(task) and max_grid_cells(task) <= max_cells:
            eligible.append(path)
    rng = random.Random(selection_seed)
    rng.shuffle(eligible)
    end = offset + limit
    if end > len(eligible):
        raise ValueError(f"requested slice {offset}:{end}, only {len(eligible)} eligible tasks")
    return eligible[offset:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--offset", type=int, choices=[0, 30], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--selection-seed", type=int, default=20260915)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--wave-channels", type=int, default=8)
    parser.add_argument("--real-channels", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    paths = choose_task_slice(
        args.data_dir, args.limit, args.offset, args.selection_seed, args.max_cells
    )
    print("task_ids", ",".join(p.stem for p in paths), flush=True)

    probe_wave = AblatedWaveARC(args.wave_channels, args.steps, args.seed, "linear")
    probe_real = RealQuadraticARC(args.real_channels, args.steps, args.seed)
    print("linear_wave_parameters", count_parameters(probe_wave), flush=True)
    print("real_quadratic_parameters", count_parameters(probe_real), flush=True)

    rows: list[dict] = []
    for path in paths:
        task = load_task(path)
        for name, model in (
            ("linear_wave", AblatedWaveARC(args.wave_channels, args.steps, args.seed, "linear")),
            ("real_quadratic", RealQuadraticARC(args.real_channels, args.steps, args.seed)),
        ):
            train_score, test_score, elapsed = train_model(model, task, args.iterations, args.lr)
            row = {
                "task_id": path.stem,
                "slice_offset": args.offset,
                "model": name,
                "seed": args.seed,
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
        "task_id", "slice_offset", "model", "seed", "parameters",
        "train_exact", "train_pixel", "test_exact", "test_pixel", "seconds",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for name in ("linear_wave", "real_quadratic"):
        rr = [r for r in rows if r["model"] == name]
        exact = sum(float(r["test_exact"]) == 1.0 for r in rr)
        train_fit = sum(float(r["train_exact"]) == 1.0 for r in rr)
        mean_pixel = sum(float(r["test_pixel"]) for r in rr) / len(rr)
        print(
            "SUMMARY", "offset", args.offset, name,
            "exact", exact, "/", len(rr),
            "train_fit", train_fit, "/", len(rr),
            "mean_pixel", round(mean_pixel, 6),
            flush=True,
        )


if __name__ == "__main__":
    main()
