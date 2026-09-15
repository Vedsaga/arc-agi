#!/usr/bin/env python3
"""Causal ablations for the frozen Wave ARC fair-30 experiment.

Modes:
  no_phase : keep fixed complex propagation/interference and amplitude saturation,
             but remove the trainable intensity-dependent phase rotation.
  linear   : keep fixed complex propagation/interference and square-law detector
             readout, but remove both recurrent phase rotation and amplitude
             saturation.  The recurrent field dynamics are then complex-linear
             apart from the final quadratic detector.

Task selection, optimizer, steps, and training budget are identical to the
pre-registered fair-30 experiment.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from wave_arc_kill_test import (
    NeighborWaveARC,
    choose_tasks,
    count_parameters,
    load_task,
    train_model,
)


class AblatedWaveARC(NeighborWaveARC):
    def __init__(self, channels: int = 8, steps: int = 4, seed: int = 0, mode: str = "no_phase"):
        super().__init__(channels=channels, steps=steps, seed=seed)
        if mode not in {"no_phase", "linear"}:
            raise ValueError(mode)
        self.mode = mode

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = grid[None]
        er = self.embed_r[x].permute(0, 3, 1, 2)
        ei = self.embed_i[x].permute(0, 3, 1, 2)
        hr, hi = er, ei

        for _ in range(self.steps):
            pr, pi = self._propagate(hr, hi)
            rr = torch.einsum("bchw,dc->bdhw", pr, self.mix_r) - torch.einsum("bchw,dc->bdhw", pi, self.mix_i)
            ri = torch.einsum("bchw,dc->bdhw", pr, self.mix_i) + torch.einsum("bchw,dc->bdhw", pi, self.mix_r)

            if self.mode == "no_phase":
                amp2 = rr.square() + ri.square()
                saturation = torch.rsqrt(1.0 + 0.15 * amp2)
                hr = saturation * rr + 0.25 * er
                hi = saturation * ri + 0.25 * ei
            else:  # complex-linear recurrent field dynamics
                hr = rr + 0.25 * er
                hi = ri + 0.25 * ei

        fr = torch.einsum("bchw,kc->bkhw", hr, self.out_r) - torch.einsum("bchw,kc->bkhw", hi, self.out_i)
        fi = torch.einsum("bchw,kc->bkhw", hr, self.out_i) + torch.einsum("bchw,kc->bkhw", hi, self.out_r)
        power = fr.square() + fi.square() + 1e-7
        return torch.log(power)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no_phase", "linear"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--selection-seed", type=int, default=20260915)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    paths = choose_tasks(args.data_dir, args.limit, args.selection_seed, args.max_cells, None)
    rows = []
    for path in paths:
        task = load_task(path)
        model = AblatedWaveARC(args.channels, args.steps, args.seed, args.mode)
        train_score, test_score, elapsed = train_model(model, task, args.iterations, args.lr)
        row = {
            "task_id": path.stem,
            "model": args.mode,
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
    fields = ["task_id", "model", "seed", "parameters", "train_exact", "train_pixel", "test_exact", "test_pixel", "seconds"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    exact = sum(float(r["test_exact"]) == 1.0 for r in rows)
    train_fit = sum(float(r["train_exact"]) == 1.0 for r in rows)
    mean_pixel = sum(float(r["test_pixel"]) for r in rows) / len(rows)
    print("SUMMARY", args.mode, "seed", args.seed, "exact", exact, "/", len(rows), "train_fit", train_fit, "/", len(rows), "mean_pixel", mean_pixel, flush=True)


if __name__ == "__main__":
    main()
