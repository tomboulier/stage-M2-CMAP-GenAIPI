#!/usr/bin/env python3
"""Contrôle de budget : le plancher observé est-il un plancher de *données* ou de *calcul* ?

Le balayage principal fixe le budget d'adaptation à 900 pas pour tous les
points, ce qui est la bonne convention pour isoler l'effet de ``n``. Mais elle
laisse une ambiguïté d'interprétation majeure :

- si la courbe du bras pré-entraîné plafonne, est-ce parce que les données
  supplémentaires n'apportent plus rien (**plancher de données**, `Q∞`), ou
  parce que 900 pas ne suffisent pas à exploiter ces données (**plancher de
  calcul**) ? Dans le premier cas la recommandation est « changez de méthode »,
  dans le second « entraînez plus longtemps ». Ce sont des conclusions opposées ;
- si le bras appris à partir de zéro ne s'améliore pas avec ``n``, est-ce une
  propriété du problème ou simplement le fait que 900 pas ne suffisent pas à
  apprendre une distribution d'images depuis une initialisation aléatoire ?

Ce script relance quelques conditions avec un budget **quadruplé** (3 600 pas).
C'est le contrôle minimal qui permet de distinguer les deux lectures, et il
faut le faire avant d'écrire quoi que ce soit sur `Q∞`.

    OMP_NUM_THREADS=1 python scripts/run_budget_control.py --shard 0 --n-shards 4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from fewshotgen.pipeline import ExperimentConfig, append_jsonl, load_jsonl, run_one  # noqa: E402
from run_pilot import RUNS_JSONL, load_prepared  # noqa: E402

#: 4x le budget du balayage principal.
STEPS = 3600
ARMS = ("pretrained_4x", "scratch_4x")
N_VALUES = (16, 256)
SEEDS = (0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(1)

    cfg = ExperimentConfig()
    cfg.finetune_steps = STEPS
    fx, pretrained_path = load_prepared(cfg)

    runs = [(n, s, a) for a in ARMS for n in N_VALUES for s in SEEDS]
    runs = runs[args.shard :: args.n_shards]
    done = {(r["n"], r["seed"], r["arm"]) for r in load_jsonl(RUNS_JSONL)}
    todo = [r for r in runs if r not in done]
    print(f"[budget shard {args.shard}] {len(todo)} exécutions à {STEPS} pas", flush=True)

    for n, seed, arm in todo:
        # `run_one` choisit l'initialisation sur le préfixe du nom du bras :
        # 'pretrained_4x' repart du modèle pré-entraîné, 'scratch_4x' de zéro.
        rec = run_one(cfg, fx, n, seed, arm, pretrained_path, verbose=True)
        rec["finetune_steps"] = STEPS
        append_jsonl(RUNS_JSONL, rec)


if __name__ == "__main__":
    main()
