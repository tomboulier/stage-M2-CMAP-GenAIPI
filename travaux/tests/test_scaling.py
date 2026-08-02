"""Validation de l'estimation de la loi d'échelle et de ``n*``.

Deux niveaux de validation :

1. *retrouver des paramètres connus* : on simule une courbe
   ``q = q_inf + a n^-alpha`` bruitée et on vérifie que l'ajustement récupère
   les paramètres et que l'intervalle de confiance bootstrap couvre ``n*`` ;
2. *cas où la réponse est mathématiquement connue* : la mesure empirique de
   ``n`` points a une MMD^2 à la loi mère qui décroît exactement en ``1/n``.
   C'est un test de bout en bout : les métriques et l'ajustement doivent
   retrouver ``alpha = 1``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fewshotgen.metrics import median_bandwidth, mmd2_unbiased  # noqa: E402
from fewshotgen.scaling import (  # noqa: E402
    bootstrap_scaling,
    data_multiplier,
    fit_power_law,
)


class TestFit:
    def test_recovers_known_parameters(self):
        n = np.array([4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
        q_inf, a, alpha = 0.02, 3.0, 0.7
        rng = np.random.default_rng(0)
        q = (q_inf + a * n**-alpha) * np.exp(rng.normal(0, 0.03, len(n)))
        f = fit_power_law(n, q)
        assert abs(f.alpha - alpha) < 0.12
        assert abs(f.q_inf - q_inf) < 0.02
        assert abs(np.log(f.a / a)) < 0.4

    def test_recovers_pure_power_law(self):
        """Sans plancher, l'ajustement doit retrouver la pente log-log."""
        n = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
        q = 5.0 * n**-0.5
        f = fit_power_law(n, q, allow_floor=False)
        assert abs(f.alpha - 0.5) < 0.02

    def test_n_star_inversion(self):
        n = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
        q = 0.01 + 2.0 * n**-0.8
        f = fit_power_law(n, q)
        tau = 0.05
        n_star = f.n_star(tau)
        assert f.predict([n_star])[0] == pytest.approx(tau, rel=1e-3)

    def test_unreachable_threshold_returns_inf(self):
        """Si le seuil est sous le plancher, aucune quantité de données ne suffit."""
        n = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
        q = 0.10 + 2.0 * n**-0.8
        f = fit_power_law(n, q)
        assert np.isinf(f.n_star(0.05))
        assert np.isfinite(f.n_star(0.5))


class TestBootstrap:
    def test_ci_covers_truth(self):
        """L'IC bootstrap doit couvrir le n* vrai dans la grande majorité des cas."""
        n_values = np.array([4, 8, 16, 32, 64, 128, 256])
        q_inf, a, alpha, tau = 0.01, 2.0, 0.6, 0.06
        true_n_star = (a / (tau - q_inf)) ** (1 / alpha)
        covered = 0
        n_trials = 20
        for trial in range(n_trials):
            rng = np.random.default_rng(trial)
            q_by_run = {
                int(n): list((q_inf + a * n**-alpha) * np.exp(rng.normal(0, 0.10, 3)))
                for n in n_values
            }
            res = bootstrap_scaling(n_values, q_by_run, tau, n_boot=300, seed=trial)
            lo, hi = res.n_star_ci
            if np.isfinite(lo) and lo <= true_n_star <= hi:
                covered += 1
        assert covered >= 0.7 * n_trials, f"couverture {covered}/{n_trials}"

    def test_reports_unreachability(self):
        """Quand le seuil est sous le plancher, le bootstrap doit le signaler."""
        n_values = np.array([4, 8, 16, 32, 64, 128, 256])
        rng = np.random.default_rng(1)
        q_by_run = {
            int(n): list((0.10 + 2.0 * n**-0.6) * np.exp(rng.normal(0, 0.05, 3)))
            for n in n_values
        }
        res = bootstrap_scaling(n_values, q_by_run, tau=0.05, n_boot=300, seed=0)
        assert res.p_unreachable > 0.5


class TestDataMultiplier:
    def test_horizontal_shift(self):
        """Deux courbes de même pente décalées d'un facteur k en n."""
        n = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
        alpha, k = 0.6, 10.0
        fast = fit_power_law(n, 1.0 * n**-alpha, allow_floor=False)
        slow = fit_power_law(n, 1.0 * (n / k) ** -alpha, allow_floor=False)
        assert data_multiplier(fast, slow, q_level=0.1) == pytest.approx(k, rel=0.05)


class TestEmpiricalMeasureEndToEnd:
    def test_alpha_one_for_empirical_measure(self):
        """Test de bout en bout avec une réponse connue analytiquement.

        Le « générateur » est ici la mesure empirique de n points : on sait que
        E[MMD^2(mesure empirique de n points, loi mère)] = C / n exactement.
        La chaîne complète (métrique -> agrégation -> ajustement) doit donc
        retrouver alpha = 1 et un plancher nul.
        """
        d = 4
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(4000, d))
        bw = median_bandwidth(ref, seed=0)
        n_values = np.array([4, 8, 16, 32, 64, 128, 256])
        q_by_run = {}
        for n in n_values:
            vals = []
            for rep in range(8):
                r2 = np.random.default_rng(1000 + 17 * rep + n)
                train = r2.normal(size=(n, d))
                # rééchantillonnage avec remise = mesure empirique
                fake = train[r2.integers(0, n, 1500)]
                vals.append(mmd2_unbiased(fake, ref, bw))
            q_by_run[int(n)] = vals
        res = bootstrap_scaling(n_values, q_by_run, tau=0.01, n_boot=200, seed=0)
        assert abs(res.fit.alpha - 1.0) < 0.2, f"alpha={res.fit.alpha:.3f}"
        assert res.fit.q_inf < 1e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
