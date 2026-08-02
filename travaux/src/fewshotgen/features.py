"""Espace de représentation dans lequel les métriques distributionnelles sont calculées.

Pourquoi ne pas utiliser InceptionV3 (le choix par défaut de la FID) ?

En imagerie médicale, InceptionV3 est entraîné sur ImageNet : ses activations
sont peu sensibles aux structures qui portent l'information clinique (une
hémorragie de 2 cm change peu la FID ImageNet) et très sensibles à des
propriétés de texture sans intérêt. C'est un biais documenté, et c'est la
première recommandation méthodologique de ce travail : **calculer les
métriques dans un espace de représentation spécifique au domaine**.

Ici, l'encodeur est un petit CNN entraîné, sur un vivier *disjoint* des jeux
d'entraînement et d'évaluation, à régresser des attributs anatomiques
(nombre de lésions, charge lésionnelle, taille de la tête, excentricité).
Dans le cadre réel du stage, le remplaçant naturel est l'encodeur d'un réseau
de segmentation de lésions traumatiques déjà entraîné -- typiquement
BLAST-CT, déjà cité dans le sujet.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .phantom import sample_dataset

FEATURE_DIM = 64


class FeatureNet(nn.Module):
    """Petit CNN : encodeur -> représentation 64-d -> têtes de régression."""

    def __init__(self, dim: int = FEATURE_DIM, n_targets: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.proj = nn.Linear(128, dim)
        self.head = nn.Linear(dim, n_targets)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        # Concaténation moyenne + max : la moyenne capte la composition
        # globale, le max la présence d'une structure focale (une lésion).
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.proj(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.silu(self.features(x)))


TARGETS = ("n_lesions", "lesion_area", "head_area", "midline_shift")


def _cache_path(size: int, seed: int, cache_dir: str) -> str:
    key = hashlib.sha1(f"featnet-v1-{size}-{seed}".encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"featnet-{key}.pt")


def train_feature_net(
    size: int = 40,
    n_pool: int = 6000,
    steps: int = 1500,
    seed: int = 777,
    cache_dir: str = "results/cache",
    verbose: bool = False,
) -> FeatureNet:
    """Entraîne (ou recharge) l'encodeur de référence.

    Le vivier d'entraînement mélange les deux domaines et utilise une graine
    dédiée (``seed=777``), disjointe de celles des expériences, pour qu'aucune
    image d'évaluation n'ait servi à construire l'espace de mesure.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(size, seed, cache_dir)
    net = FeatureNet()
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        return net

    xs, ys = [], []
    for i, dom in enumerate(("source", "target")):
        imgs, attrs = sample_dataset(n_pool // 2, dom, size, seed=seed + i)
        xs.append(imgs)
        ys.append(np.stack([attrs[k] for k in TARGETS], axis=1))
    X = torch.from_numpy(np.concatenate(xs)).float()
    Y = torch.from_numpy(np.concatenate(ys)).float()
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)

    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for step in range(steps):
        idx = torch.randint(0, X.shape[0], (128,), generator=gen)
        loss = F.mse_loss(net(X[idx]), Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (step + 1) % 250 == 0:
            print(f"  featnet step {step + 1}/{steps} loss={float(loss.detach()):.4f}")
    net.eval()
    torch.save(net.state_dict(), path)
    return net


@torch.no_grad()
def embed(net: FeatureNet, images: np.ndarray | torch.Tensor, batch: int = 512) -> np.ndarray:
    """Projette un lot d'images ``(n, 1, H, W)`` dans l'espace de représentation."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    images = images.float()
    net.eval()
    out = [net.features(images[i : i + batch]).numpy() for i in range(0, len(images), batch)]
    return np.concatenate(out, axis=0).astype(np.float64)
