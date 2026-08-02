"""Orchestration de l'expérience pilote.

Plan d'expérience
-----------------
On compare trois *bras* d'adaptation, à budget de calcul d'adaptation
strictement identique :

``pretrained``
    initialisation par le modèle pré-entraîné sur le domaine source ;
``scratch``
    initialisation aléatoire (contrôle négatif) ;
``pretrained_aug``
    idem ``pretrained``, plus une augmentation (symétrie gauche/droite +
    bruit léger), le levier le plus économique en pratique.

Pour chaque bras et chaque taille ``n``, ``S`` répétitions indépendantes
(graine d'entraînement *et* tirage du sous-échantillon d'apprentissage).

Points de plan d'expérience qui comptent, et qui sont la principale source
d'erreurs dans la littérature sur ce sujet :

1. **budget de calcul constant** : le nombre de pas d'optimisation ne dépend
   pas de ``n``. Sinon la courbe mesure un mélange de « plus de données » et
   « plus de calcul » ;
2. **jeux d'évaluation gelés** : le même jeu de référence, la même largeur de
   bande de noyau et le même nombre d'échantillons générés pour tous les
   points de la courbe. Sinon le biais des estimateurs varie le long de la
   courbe ;
3. **sous-échantillons non emboîtés** : les ``n`` images sont retirées
   indépendamment à chaque répétition. Des sous-échantillons emboîtés
   (n=4 inclus dans n=8 inclus dans n=16...) corrèlent les points et font
   sous-estimer les intervalles de confiance ;
4. **jeu de mémorisation séparé** : le ``holdout`` qui calibre l'audit de
   nouveauté est disjoint du jeu de référence, sans quoi l'audit et la mesure
   de qualité partagent leur bruit d'échantillonnage.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .ddpm import Diffusion, TrainConfig, train
from .features import embed, train_feature_net
from .metrics import evaluate, median_bandwidth, mmd2_unbiased
from .phantom import sample_dataset
from .unet import UNet


@dataclass
class ExperimentConfig:
    size: int = 32
    base: int = 24
    mults: tuple[int, ...] = (1, 2, 2)
    diffusion_T: int = 1000

    # Pré-entraînement sur le domaine source.
    n_source: int = 20000
    pretrain_steps: int = 5000
    pretrain_batch: int = 64
    pretrain_lr: float = 2e-4

    # Adaptation.
    finetune_steps: int = 900
    finetune_batch: int = 32
    finetune_lr: float = 3e-4

    # Évaluation.
    n_gen: int = 384
    ddim_steps: int = 30
    n_ref: int = 1024
    n_holdout: int = 512
    n_pool: int = 2048
    n_null: int = 4096

    n_values: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
    seeds: tuple[int, ...] = (0, 1, 2)
    arms: tuple[str, ...] = ("pretrained", "scratch", "pretrained_aug")

    # Graines des jeux de données, disjointes entre elles et de celles des
    # entraînements.
    seed_source: int = 1000
    seed_pool: int = 2000
    seed_ref: int = 3000
    seed_holdout: int = 4000
    seed_null: int = 5000

    out_dir: str = "results"
    cache_dir: str = "results/cache"

    def to_json(self) -> dict:
        d = asdict(self)
        d["mults"] = list(self.mults)
        d["n_values"] = list(self.n_values)
        d["seeds"] = list(self.seeds)
        d["arms"] = list(self.arms)
        return d


@dataclass
class Fixtures:
    """Ensembles de données et objets d'évaluation gelés, partagés par tous les runs."""

    cfg: ExperimentConfig
    pool: np.ndarray
    ref: np.ndarray
    holdout: np.ndarray
    feat_net: object
    bandwidth: float
    tau_null: dict[str, float] = field(default_factory=dict)


def build_fixtures(cfg: ExperimentConfig, verbose: bool = True) -> Fixtures:
    """Construit (une fois) les jeux d'évaluation, l'encodeur et le seuil ``tau``."""
    pool, _ = sample_dataset(cfg.n_pool, "target", cfg.size, seed=cfg.seed_pool)
    ref, _ = sample_dataset(cfg.n_ref, "target", cfg.size, seed=cfg.seed_ref)
    holdout, _ = sample_dataset(cfg.n_holdout, "target", cfg.size, seed=cfg.seed_holdout)

    if verbose:
        print("[fixtures] entraînement de l'encodeur de référence...")
    feat_net = train_feature_net(size=cfg.size, cache_dir=cfg.cache_dir, verbose=verbose)

    F_ref = embed(feat_net, ref)
    bandwidth = median_bandwidth(F_ref, seed=0)

    if verbose:
        print("[fixtures] calibration du seuil tau (loi nulle réel/réel)...")
    tau_null = calibrate_tau(cfg, feat_net, bandwidth, verbose=verbose)
    return Fixtures(cfg, pool, ref, holdout, feat_net, bandwidth, tau_null)


def calibrate_tau(
    cfg: ExperimentConfig, feat_net, bandwidth: float, n_rep: int = 200, verbose: bool = False
) -> dict[str, float]:
    """Loi nulle de la statistique MMD^2 entre deux échantillons *réels* disjoints.

    Le seuil ``tau_95`` est le quantile 95 % de cette loi : atteindre
    ``MMD^2 <= tau_95`` signifie que le jeu généré n'est pas distinguable d'un
    jeu réel par ce test, aux effectifs d'évaluation utilisés. C'est
    l'équivalent exact d'une marge de non-infériorité, et cela rend ``n*``
    interprétable au lieu d'arbitraire.
    """
    big, _ = sample_dataset(cfg.n_null, "target", cfg.size, seed=cfg.seed_null)
    F = embed(feat_net, big)
    rng = np.random.default_rng(12345)
    stats = []
    for _ in range(n_rep):
        idx = rng.permutation(len(F))
        a = F[idx[: cfg.n_gen]]
        b = F[idx[cfg.n_gen : cfg.n_gen + cfg.n_ref]]
        stats.append(mmd2_unbiased(a, b, bandwidth))
    stats = np.asarray(stats)
    out = {
        "tau_95": float(np.quantile(stats, 0.95)),
        "tau_99": float(np.quantile(stats, 0.99)),
        "null_mean": float(stats.mean()),
        "null_sd": float(stats.std(ddof=1)),
    }
    if verbose:
        print(f"[fixtures] loi nulle : moyenne={out['null_mean']:.2e} "
              f"tau_95={out['tau_95']:.2e}")
    return out


def make_model(cfg: ExperimentConfig) -> UNet:
    return UNet(in_ch=1, base=cfg.base, mults=tuple(cfg.mults))


def pretrain(cfg: ExperimentConfig, verbose: bool = True) -> str:
    """Pré-entraîne le modèle sur le domaine source ; renvoie le chemin du point de contrôle."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = os.path.join(
        cfg.cache_dir,
        f"pretrained-s{cfg.size}-b{cfg.base}-n{cfg.n_source}-t{cfg.pretrain_steps}.pt",
    )
    if os.path.exists(path):
        return path
    if verbose:
        print(f"[pretrain] {cfg.n_source} images source, {cfg.pretrain_steps} pas...")
    X, _ = sample_dataset(cfg.n_source, "source", cfg.size, seed=cfg.seed_source)
    data = torch.from_numpy(X)
    model = make_model(cfg)
    diff = Diffusion(cfg.diffusion_T)
    tcfg = TrainConfig(
        steps=cfg.pretrain_steps,
        batch_size=cfg.pretrain_batch,
        lr=cfg.pretrain_lr,
        warmup=200,
        ema_decay=0.999,
    )
    t0 = time.time()
    ema, _ = train(model, data, diff, tcfg, seed=cfg.seed_source)
    torch.save(ema.state_dict(), path)
    if verbose:
        print(f"[pretrain] terminé en {time.time() - t0:.0f} s -> {path}")
    return path


def subsample(pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Tire ``n`` images du vivier, sans emboîtement entre répétitions."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    return pool[idx]


def run_one(
    cfg: ExperimentConfig,
    fx: Fixtures,
    n: int,
    seed: int,
    arm: str,
    pretrained_path: str | None,
    verbose: bool = False,
) -> dict:
    """Exécute une condition expérimentale complète et renvoie ses métriques."""
    t0 = time.time()
    # Graine du sous-échantillon : dépend de (n, seed) mais pas du bras, afin
    # que les trois bras voient exactement les mêmes images. La comparaison
    # entre bras est donc appariée, ce qui réduit fortement sa variance.
    train_imgs = subsample(fx.pool, n, seed=90000 + 97 * seed + n)
    data = torch.from_numpy(train_imgs)

    model = make_model(cfg)
    if arm.startswith("pretrained"):
        if pretrained_path is None:
            raise ValueError("bras pré-entraîné demandé sans point de contrôle")
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

    tcfg = TrainConfig(
        steps=cfg.finetune_steps,
        batch_size=cfg.finetune_batch,
        lr=cfg.finetune_lr,
        warmup=50,
        ema_decay=0.995,
        p_flip=0.5 if arm.endswith("_aug") else 0.0,
        aug_noise=0.02 if arm.endswith("_aug") else 0.0,
    )
    ema, _ = train(model, data, Diffusion(cfg.diffusion_T), tcfg, seed=seed)

    diff = Diffusion(cfg.diffusion_T)
    gen = torch.Generator().manual_seed(777_000 + seed)
    fake = diff.ddim_sample(
        ema, cfg.n_gen, (1, cfg.size, cfg.size), n_steps=cfg.ddim_steps, generator=gen
    ).numpy()

    ev = evaluate(
        fake_images=fake,
        real_ref_images=fx.ref,
        train_images=train_imgs,
        holdout_images=fx.holdout,
        feat_net=fx.feat_net,
        bandwidth=fx.bandwidth,
    )
    # Conserve une planche d'échantillons : les métriques ne remplacent pas
    # l'inspection visuelle, qui reste le premier réflexe d'un radiologue.
    grid_dir = os.path.join(cfg.out_dir, "samples")
    os.makedirs(grid_dir, exist_ok=True)
    np.save(os.path.join(grid_dir, f"{arm}_n{n}_s{seed}.npy"), fake[:16].astype(np.float16))

    rec = {"n": n, "seed": seed, "arm": arm, "seconds": round(time.time() - t0, 1)}
    rec.update(ev.as_dict())
    if verbose:
        print(
            f"[run] arm={arm:16s} n={n:4d} seed={seed} "
            f"mmd2={rec['mmd2']:.3e} cov={rec['coverage']:.2f} "
            f"auth={rec['authenticity']:.2f} w1={rec['w1_mean']:.2f} "
            f"({rec['seconds']:.0f}s)"
        )
    return rec


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plan_runs(cfg: ExperimentConfig) -> list[tuple[int, int, str]]:
    """Liste ordonnée des conditions à exécuter.

    L'ordre est *par bras* : si l'exécution est interrompue, les deux bras
    principaux (``pretrained``, ``scratch``) sont complets avant que le bras
    d'augmentation ne commence, et les résultats restent exploitables.
    """
    runs = []
    for arm in cfg.arms:
        for n in cfg.n_values:
            for seed in cfg.seeds:
                runs.append((n, seed, arm))
    return runs
