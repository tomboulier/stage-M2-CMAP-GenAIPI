#!/usr/bin/env python3
"""Exécute l'expérience pilote « combien d'images faut-il ? ».

Usage
-----
    # 1. préparation (une seule fois) : encodeur, seuil tau, pré-entraînement
    python scripts/run_pilot.py --prepare

    # 2. balayage, réparti sur N processus mono-thread (bien plus rapide que
    #    1 processus multi-thread sur ces petits tenseurs)
    for i in 0 1 2 3; do
        OMP_NUM_THREADS=1 python scripts/run_pilot.py --shard $i --n-shards 4 &
    done; wait

Les résultats sont ajoutés au fur et à mesure dans ``results/runs.jsonl`` :
une interruption ne perd que le run en cours, et une relance reprend là où
l'exécution s'était arrêtée.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from fewshotgen.pipeline import (  # noqa: E402
    ExperimentConfig,
    Fixtures,
    append_jsonl,
    build_fixtures,
    load_jsonl,
    plan_runs,
    pretrain,
    run_one,
)


FIXTURES_JSON = "results/fixtures.json"
RUNS_JSONL = "results/runs.jsonl"


def prepare(cfg: ExperimentConfig) -> None:
    """Construit tout ce qui doit être partagé et identique entre les runs."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    fx = build_fixtures(cfg, verbose=True)
    path = pretrain(cfg, verbose=True)
    with open(FIXTURES_JSON, "w") as f:
        json.dump(
            {
                "config": cfg.to_json(),
                "bandwidth": fx.bandwidth,
                "tau_null": fx.tau_null,
                "pretrained_path": path,
            },
            f,
            indent=2,
        )
    print(f"[prepare] écrit {FIXTURES_JSON}")


def load_prepared(cfg: ExperimentConfig) -> tuple[Fixtures, str]:
    """Reconstruit les fixtures dans un worker, sans recalculer tau."""
    from fewshotgen.features import train_feature_net
    from fewshotgen.phantom import sample_dataset

    with open(FIXTURES_JSON) as f:
        meta = json.load(f)
    pool, _ = sample_dataset(cfg.n_pool, "target", cfg.size, seed=cfg.seed_pool)
    ref, _ = sample_dataset(cfg.n_ref, "target", cfg.size, seed=cfg.seed_ref)
    holdout, _ = sample_dataset(cfg.n_holdout, "target", cfg.size, seed=cfg.seed_holdout)
    feat_net = train_feature_net(size=cfg.size, cache_dir=cfg.cache_dir)
    fx = Fixtures(
        cfg=cfg,
        pool=pool,
        ref=ref,
        holdout=holdout,
        feat_net=feat_net,
        bandwidth=meta["bandwidth"],
        tau_null=meta["tau_null"],
    )
    return fx, meta["pretrained_path"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--finetune-steps", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = ExperimentConfig()
    if args.finetune_steps:
        cfg.finetune_steps = args.finetune_steps

    if args.prepare:
        prepare(cfg)
        return

    fx, pretrained_path = load_prepared(cfg)
    runs = plan_runs(cfg)[args.shard :: args.n_shards]
    done = {(r["n"], r["seed"], r["arm"]) for r in load_jsonl(RUNS_JSONL)}
    todo = [r for r in runs if r not in done]
    print(f"[shard {args.shard}] {len(todo)}/{len(runs)} runs à exécuter", flush=True)

    t0 = time.time()
    for i, (n, seed, arm) in enumerate(todo):
        rec = run_one(cfg, fx, n, seed, arm, pretrained_path, verbose=True)
        append_jsonl(RUNS_JSONL, rec)
        elapsed = time.time() - t0
        print(
            f"[shard {args.shard}] {i + 1}/{len(todo)} — "
            f"{elapsed / 60:.1f} min écoulées, "
            f"~{elapsed / (i + 1) * (len(todo) - i - 1) / 60:.1f} min restantes",
            flush=True,
        )


if __name__ == "__main__":
    main()
