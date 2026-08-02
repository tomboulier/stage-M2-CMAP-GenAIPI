"""UNet 2D conditionné par le temps, volontairement compact.

L'architecture reprend celle de Ho et al. (DDPM) / Nichol & Dhariwal en la
réduisant à ce qui tient sur CPU : deux niveaux de résolution, une attention
au goulot d'étranglement, ~1 M de paramètres. Elle n'a pas vocation à être
compétitive, mais à être *représentative* : ce sont les mêmes blocs que dans
les UNet de MONAI/MAISI, à l'échelle près.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Plongement sinusoïdal du pas de bruit (Vaswani et al.)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Initialisation à zéro de la dernière convolution : le bloc démarre
        # comme l'identité, ce qui stabilise nettement le fine-tuning à
        # très peu de données.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        att = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (v @ att.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """UNet epsilon-prédictif.

    Parameters
    ----------
    base : largeur du premier niveau.
    mults : multiplicateurs de largeur par niveau de résolution.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        mults: tuple[int, ...] = (1, 2, 2),
        t_dim: int | None = None,
    ):
        super().__init__()
        t_dim = t_dim or base * 4
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.base = base
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        chans = [base * m for m in mults]
        self.downs = nn.ModuleList()
        self.downsample = nn.ModuleList()
        prev = base
        skip_chans = [base]
        for i, ch in enumerate(chans):
            self.downs.append(ResBlock(prev, ch, t_dim))
            skip_chans.append(ch)
            prev = ch
            if i < len(chans) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsample.append(nn.Identity())

        self.mid1 = ResBlock(prev, prev, t_dim)
        self.mid_attn = SelfAttention(prev)
        self.mid2 = ResBlock(prev, prev, t_dim)

        self.ups = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.ups.append(ResBlock(prev + skip_chans[i + 1], ch, t_dim))
            prev = ch
            if i > 0:
                self.upsample.append(nn.Upsample(scale_factor=2, mode="nearest"))
            else:
                self.upsample.append(nn.Identity())

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.stem(x)
        skips = [h]
        for block, down in zip(self.downs, self.downsample):
            h = block(h, temb)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for block, up in zip(self.ups, self.upsample):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), temb)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
