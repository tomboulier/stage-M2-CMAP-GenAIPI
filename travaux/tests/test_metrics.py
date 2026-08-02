"""Validation des métriques sur des cas où la réponse est connue.

Une métrique d'évaluation générative est un instrument de mesure : avant de
s'en servir pour conclure quoi que ce soit sur un modèle, il faut vérifier
qu'elle répond correctement sur des situations construites. C'est
particulièrement vrai des métriques utilisées dans ce travail, dont les
estimateurs ont des biais bien documentés.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fewshotgen.metrics import (  # noqa: E402
    attribute_discrepancy,
    frechet_distance,
    median_bandwidth,
    memorization,
    mmd2_unbiased,
    precision_recall_density_coverage,
)
from fewshotgen.phantom import sample_dataset  # noqa: E402


def _gauss(n, d, seed, shift=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(shift, scale, size=(n, d))


class TestMMD:
    def test_unbiased_under_null(self):
        """Sous H0, E[MMD^2] = 0 : la moyenne sur des répétitions doit être ~0."""
        d = 8
        bw = median_bandwidth(_gauss(500, d, 0))
        vals = [
            mmd2_unbiased(_gauss(200, d, 100 + i), _gauss(300, d, 900 + i), bw)
            for i in range(60)
        ]
        m, s = np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals))
        assert abs(m) < 4 * s, f"biais détecté sous H0 : {m:.2e} +- {s:.2e}"

    def test_no_sample_size_bias(self):
        """L'espérance ne doit pas dépendre du nombre d'échantillons.

        C'est la propriété qui rend la MMD^2 sans biais utilisable le long
        d'une courbe qualité/n, contrairement à la FID.
        """
        d = 8
        bw = median_bandwidth(_gauss(500, d, 0))
        means = []
        for n in (50, 200, 800):
            vals = [
                mmd2_unbiased(_gauss(n, d, 5000 + i), _gauss(400, d, 7000 + i), bw)
                for i in range(40)
            ]
            means.append(np.mean(vals))
        assert max(abs(np.diff(means))) < 5e-3, f"biais dépendant de n : {means}"

    def test_detects_shift(self):
        d = 8
        bw = median_bandwidth(_gauss(500, d, 0))
        null = mmd2_unbiased(_gauss(300, d, 1), _gauss(300, d, 2), bw)
        alt = mmd2_unbiased(_gauss(300, d, 1), _gauss(300, d, 2, shift=1.0), bw)
        assert alt > 10 * abs(null) and alt > 0.05

    def test_fid_is_biased_downward_in_n(self):
        """Contre-exemple documentant pourquoi la FID brute est piégeuse ici.

        Entre deux échantillons de la *même* loi, la FID devrait valoir 0 ;
        elle décroît en réalité avec le nombre d'échantillons. Deux modèles
        évalués avec des effectifs différents ne sont donc pas comparables.
        """
        d = 8
        fid_small = np.mean(
            [frechet_distance(_gauss(40, d, i), _gauss(400, d, 500 + i)) for i in range(20)]
        )
        fid_large = np.mean(
            [frechet_distance(_gauss(400, d, i), _gauss(400, d, 500 + i)) for i in range(20)]
        )
        assert fid_small > 3 * fid_large > 0


class TestPRDC:
    def test_identical_distributions(self):
        d = 8
        real, fake = _gauss(400, d, 1), _gauss(400, d, 2)
        r = precision_recall_density_coverage(real, fake, k=5)
        assert r["precision"] > 0.8 and r["recall"] > 0.8
        assert r["coverage"] > 0.8

    def test_mode_collapse_is_caught(self):
        """Un générateur qui ne produit qu'un mode : précision haute, rappel bas."""
        d = 8
        real = _gauss(400, d, 1)
        fake = _gauss(400, d, 2, scale=0.05)  # concentré près de 0
        r = precision_recall_density_coverage(real, fake, k=5)
        assert r["precision"] > 0.5
        assert r["recall"] < 0.3
        assert r["coverage"] < 0.3

    def test_out_of_support_is_caught(self):
        d = 8
        real = _gauss(400, d, 1)
        fake = _gauss(400, d, 2, shift=6.0)
        r = precision_recall_density_coverage(real, fake, k=5)
        assert r["precision"] < 0.1


class TestMemorization:
    def test_copies_are_detected(self):
        """Un « générateur » qui recopie son jeu d'entraînement doit être démasqué."""
        d = 8
        train = _gauss(64, d, 1)
        holdout = _gauss(256, d, 2)
        fake = train[np.random.default_rng(3).integers(0, 64, 200)]  # copies exactes
        m = memorization(train, fake, holdout)
        assert m["nn_ratio"] < 0.05
        assert m["copy_rate"] > 0.95
        assert m["authenticity"] < 0.05

    def test_genuine_generalization(self):
        d = 8
        train = _gauss(64, d, 1)
        holdout = _gauss(256, d, 2)
        fake = _gauss(200, d, 3)
        m = memorization(train, fake, holdout)
        assert 0.7 < m["nn_ratio"] < 1.4
        assert m["copy_rate"] < 0.05
        assert m["authenticity"] > 0.85

    def test_copies_can_have_excellent_mmd(self):
        """Le point central de la méthodologie, sous forme de test.

        Un générateur qui rééchantillonne son jeu d'entraînement obtient une
        MMD^2 excellente dès que ce jeu est assez grand : la seule métrique
        d'adéquation distributionnelle ne peut pas détecter la mémorisation.
        L'audit de nouveauté n'est donc pas un complément optionnel.
        """
        d = 8
        bw = median_bandwidth(_gauss(500, d, 0))
        train = _gauss(256, d, 1)
        ref = _gauss(500, d, 2)
        holdout = _gauss(300, d, 4)
        fake = train[np.random.default_rng(3).integers(0, 256, 300)]
        assert abs(mmd2_unbiased(fake, ref, bw)) < 0.02  # « très bon » score
        assert memorization(train, fake, holdout)["copy_rate"] > 0.95  # et pourtant


class TestAttributes:
    def test_separates_domains(self):
        """L'évaluation sémantique doit distinguer le domaine source du cible."""
        tgt_a, _ = sample_dataset(200, "target", 32, seed=11)
        tgt_b, _ = sample_dataset(200, "target", 32, seed=12)
        src, _ = sample_dataset(200, "source", 32, seed=13)
        same = attribute_discrepancy(tgt_a, tgt_b)["w1_mean"]
        diff = attribute_discrepancy(tgt_a, src)["w1_mean"]
        assert diff > 3 * same, f"same={same:.3f} diff={diff:.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
