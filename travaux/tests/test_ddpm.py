"""Vérification de la correction du processus de diffusion et du réseau."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fewshotgen.ddpm import Diffusion, TrainConfig, cosine_alpha_bar, train  # noqa: E402
from fewshotgen.phantom import DOMAINS, sample_dataset  # noqa: E402
from fewshotgen.unet import UNet  # noqa: E402


class TestSchedule:
    def test_monotone_and_bounded(self):
        ab = cosine_alpha_bar(200)
        assert ab[0] == pytest.approx(1.0)
        assert ab[-1] < 1e-2
        assert torch.all(ab[1:] <= ab[:-1] + 1e-8)

    def test_marginal_variance_is_preserved(self):
        """x_t doit rester de variance ~1 si x_0 l'est (diffusion VP)."""
        d = Diffusion(T=500)
        x0 = torch.randn(4000, 1, 4, 4)
        for t_val in (1, 100, 250, 499):
            t = torch.full((4000,), t_val, dtype=torch.long)
            xt = d.q_sample(x0, t, torch.randn_like(x0))
            assert abs(float(xt.var()) - 1.0) < 0.06, f"t={t_val}"


class TestDDIM:
    def test_perfect_model_recovers_the_data(self):
        """Avec un modèle de score exact, DDIM doit reconstruire x_0.

        On construit un « modèle » analytique pour la loi cible p = delta_{x*},
        pour laquelle eps*(x_t, t) = (x_t - sqrt(ab) x*) / sqrt(1 - ab). Le
        solveur doit alors converger vers x*, ce qui teste conjointement les
        signes, l'indexation du planning et la formule de mise à jour.
        """
        d = Diffusion(T=1000)
        target = torch.full((1, 2, 2), 0.4)

        class Oracle(torch.nn.Module):
            def forward(self, x, t):
                ab = d.alpha_bar[t][:, None, None, None]
                return (x - ab.sqrt() * target) / (1 - ab).sqrt()

        out = d.ddim_sample(Oracle(), 8, (1, 2, 2), n_steps=50)
        assert torch.allclose(out, target.expand(8, 1, 2, 2), atol=2e-3)

    def test_sampling_is_deterministic_given_a_generator(self):
        d = Diffusion(T=200)
        net = UNet(base=8, mults=(1, 2))
        a = d.ddim_sample(net, 4, (1, 16, 16), n_steps=5,
                          generator=torch.Generator().manual_seed(0))
        b = d.ddim_sample(net, 4, (1, 16, 16), n_steps=5,
                          generator=torch.Generator().manual_seed(0))
        assert torch.equal(a, b)


class TestUNet:
    def test_shapes(self):
        net = UNet(base=16, mults=(1, 2, 2))
        for size in (32, 40):
            x = torch.randn(3, 1, size, size)
            t = torch.randint(1, 1000, (3,))
            assert net(x, t).shape == x.shape

    def test_time_conditioning_is_effective(self):
        """La sortie doit dépendre de t (sinon le conditionnement est débranché)."""
        torch.manual_seed(0)
        net = UNet(base=16, mults=(1, 2))
        # Au démarrage le réseau est initialisé à zéro : on le perturbe.
        for p in net.parameters():
            p.data.add_(0.05 * torch.randn_like(p))
        x = torch.randn(2, 1, 16, 16)
        y1 = net(x, torch.tensor([1, 1]))
        y2 = net(x, torch.tensor([900, 900]))
        assert float((y1 - y2).abs().mean()) > 1e-4


class TestTrainingReducesLoss:
    def test_short_run_learns_something(self):
        """Contrôle de bout en bout : la perte doit baisser sur des données réelles."""
        X, _ = sample_dataset(128, "target", 32, seed=0)
        data = torch.from_numpy(X)
        net = UNet(base=16, mults=(1, 2, 2))
        d = Diffusion(T=1000)
        _, hist = train(net, data, d,
                        TrainConfig(steps=200, batch_size=32, lr=1e-3),
                        seed=0, log_every=20)
        assert hist[-1] < 0.8 * hist[0], f"perte {hist[0]:.3f} -> {hist[-1]:.3f}"


class TestPhantomDomains:
    def test_domains_differ_where_intended(self):
        """La différence source/cible doit porter sur les lésions, pas sur la tête."""
        _, a_src = sample_dataset(400, "source", 32, seed=1)
        _, a_tgt = sample_dataset(400, "target", 32, seed=2)
        assert a_tgt["n_lesions"].mean() > 8 * a_src["n_lesions"].mean()
        assert abs(a_tgt["head_area"].mean() - a_src["head_area"].mean()) < 0.02

    def test_images_are_in_range(self):
        X, _ = sample_dataset(64, "target", 32, seed=0)
        assert X.min() >= -1.0 and X.max() <= 1.0
        assert X.dtype == np.float32

    def test_reproducible(self):
        a, _ = sample_dataset(16, "target", 32, seed=42)
        b, _ = sample_dataset(16, "target", 32, seed=42)
        assert np.array_equal(a, b)

    def test_all_domains_declared(self):
        assert set(DOMAINS) == {"source", "target"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
