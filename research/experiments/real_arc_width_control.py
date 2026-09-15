#!/usr/bin/env python3
"""Frozen fair-30 control with 16 real recurrent channels.

Eight complex wave channels contain 16 real state coordinates.  This control
therefore matches the wave model's state dimensionality while giving the real
model *more* trainable parameters.  Task selection and training budget are
unchanged.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from wave_arc_fair30 import RealMatchedARC
from wave_arc_kill_test import choose_tasks, count_parameters, load_task, train_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/training"))
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--selection-seed", type=int, default=20260915)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.015)
    p.add_argument("--max-cells", type=int, default=400)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    torch.set_num_threads(1)
    paths = choose_tasks(args.data_dir, args.limit, args.selection_seed, args.max_cells, None)
    rows=[]
    for path in paths:
        task=load_task(path)
        model=RealMatchedARC(args.channels,args.steps,args.seed)
        train_score,test_score,elapsed=train_model(model,task,args.iterations,args.lr)
        row={
            "task_id":path.stem,"model":"real16","seed":args.seed,
            "parameters":count_parameters(model),
            "train_exact":train_score.exact_fraction,"train_pixel":train_score.pixel_accuracy,
            "test_exact":test_score.exact_fraction,"test_pixel":test_score.pixel_accuracy,
            "seconds":elapsed,
        }
        rows.append(row); print(row,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fields=["task_id","model","seed","parameters","train_exact","train_pixel","test_exact","test_pixel","seconds"]
    with args.output.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("SUMMARY","parameters",rows[0]["parameters"],"exact",sum(r["test_exact"]==1.0 for r in rows),"/",len(rows),"train_fit",sum(r["train_exact"]==1.0 for r in rows),"/",len(rows),"mean_pixel",sum(r["test_pixel"] for r in rows)/len(rows),flush=True)


if __name__=="__main__":
    main()
