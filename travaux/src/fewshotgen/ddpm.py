"""Modèle de diffusion (DDPM) : bruitage, entraînement, échantillonnage DDIM.

Implémentation minimale mais fidèle : planning cosinus (Nichol & Dhariwal),
paramétrisation en epsilon, échantillonneur DDIM déterministe (Song et al.),
moyenne mobile exponentielle des poids.

La moyenne mobile n'est pas un détail : en régime « peu de données », la perte
est très bruitée et l'EMA change qualitativement les échantillons. Elle est
donc traitée comme une composante du modèle, pas comme une option.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """Planning cosinus : renvoie ``alpha_bar`` de longueur ``T + 1``."""
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return (f / f[0]).float()


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup: int = 100
    ema_decay: float = 0.995
    grad_clip: float = 1.0
    #: Écart-type du bruit d'augmentation gaussien ajouté aux images
    #: d'entraînement (« noise conditioning augmentation » légère). Mis à 0
    #: par défaut : l'augmentation est étudiée séparément.
    aug_noise: float = 0.0
    #: Probabilité de retournement horizontal. Le cerveau est approximativement
    #: symétrique : c'est l'augmentation la plus défendable cliniquement, mais
    #: elle détruit l'information de latéralité de la lésion. À manipuler avec
    #: précaution -- d'où le défaut à 0.
    p_flip: float = 0.0


class EMA:
    """Moyenne mobile exponentielle des paramètres."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


class Diffusion:
    """Processus de diffusion à variance préservée."""

    def __init__(self, T: int = 1000, device: str = "cpu"):
        self.T = T
        self.device = device
        ab = cosine_alpha_bar(T).to(device)
        self.alpha_bar = ab
        self.sqrt_ab = ab.sqrt()
        self.sqrt_1mab = (1 - ab).sqrt()

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        a = self.sqrt_ab[t][:, None, None, None]
        b = self.sqrt_1mab[t][:, None, None, None]
        return a * x0 + b * noise

    def loss(self, model: nn.Module, x0: torch.Tensor, generator=None) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(1, self.T + 1, (b,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        xt = self.q_sample(x0, t, noise)
        eps = model(xt, t)
        return ((eps - noise) ** 2).mean()

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        n: int,
        shape: tuple[int, int, int],
        n_steps: int = 50,
        eta: float = 0.0,
        generator=None,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Échantillonne ``n`` images par DDIM (déterministe si ``eta = 0``)."""
        model.eval()
        outs = []
        ts = np.linspace(self.T, 0, n_steps + 1).round().astype(int)
        for start in range(0, n, batch_size):
            m = min(batch_size, n - start)
            x = torch.randn((m, *shape), device=self.device, generator=generator)
            for i in range(n_steps):
                t_cur, t_next = int(ts[i]), int(ts[i + 1])
                tt = torch.full((m,), t_cur, device=self.device, dtype=torch.long)
                eps = model(x, tt)
                ab_c = self.alpha_bar[t_cur]
                ab_n = self.alpha_bar[t_next]
                x0 = (x - (1 - ab_c).sqrt() * eps) / ab_c.sqrt()
                x0 = x0.clamp(-1, 1)
                sigma = (
                    eta
                    * ((1 - ab_n) / (1 - ab_c)).sqrt()
                    * (1 - ab_c / ab_n).clamp(min=0).sqrt()
                )
                dir_xt = (1 - ab_n - sigma**2).clamp(min=0).sqrt() * eps
                x = ab_n.sqrt() * x0 + dir_xt
                if eta > 0 and t_next > 0:
                    x = x + sigma * torch.randn(x.shape, device=self.device, generator=generator)
            outs.append(x.clamp(-1, 1).cpu())
        return torch.cat(outs, dim=0)


def train(
    model: nn.Module,
    data: torch.Tensor,
    diffusion: Diffusion,
    cfg: TrainConfig,
    seed: int = 0,
    log_every: int = 0,
) -> tuple[nn.Module, list[float]]:
    """Entraîne ``model`` sur ``data`` et renvoie ``(modèle EMA, historique)``.

    ``data`` : tenseur ``(n, C, H, W)`` dans [-1, 1]. Le nombre d'itérations
    est fixé indépendamment de ``n`` : c'est indispensable pour que la
    comparaison entre tailles d'échantillon porte sur la quantité de *données*
    et non sur la quantité de *calcul*.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=data.device).manual_seed(seed + 12345)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay)
    n = data.shape[0]
    history: list[float] = []
    running = None

    for step in range(cfg.steps):
        lr = cfg.lr * min(1.0, (step + 1) / max(cfg.warmup, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n, (min(cfg.batch_size, max(n, 1)),), generator=gen)
        x0 = data[idx].clone()
        if cfg.p_flip > 0:
            flip = torch.rand(x0.shape[0], generator=gen) < cfg.p_flip
            x0[flip] = torch.flip(x0[flip], dims=[-1])
        if cfg.aug_noise > 0:
            x0 = x0 + cfg.aug_noise * torch.randn(x0.shape, generator=gen)
        loss = diffusion.loss(model, x0, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(model)
        v = float(loss.detach())
        running = v if running is None else 0.98 * running + 0.02 * v
        if log_every and (step + 1) % log_every == 0:
            history.append(running)
    return ema.shadow, history
