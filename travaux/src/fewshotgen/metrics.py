"""Batterie d'évaluation d'un modèle génératif à faible effectif d'entraînement.

Le point méthodologique central de ce travail est le suivant : **la question
« quelle qualité atteint-on avec n images ? » n'a pas de réponse scalaire**.
Un modèle qui recopie ses n images d'entraînement obtient une fidélité
parfaite, une diversité nulle et une valeur d'usage nulle -- et, si n est
suffisant, une FID *excellente*. Toute courbe qualité/n qui ne contrôle pas la
mémorisation mesure donc en partie la capacité à mémoriser.

L'évaluation est par conséquent vectorielle, sur quatre axes :

fidélité
    Les échantillons sont-ils dans le support de la loi réelle ?
    -> ``precision``, ``density``
diversité
    Couvrent-ils tout son support ?
    -> ``recall``, ``coverage``
adéquation distributionnelle
    -> ``mmd2`` (estimateur sans biais), ``fid`` (biaisé, reporté pour
    comparaison), ``fid_inf`` (extrapolation de Chong & Forsyth)
nouveauté
    Les échantillons diffèrent-ils du jeu d'entraînement autant que de
    vraies images tenues à l'écart ?
    -> ``authenticity``, ``nn_ratio``, ``copy_rate``

S'y ajoute une évaluation *sémantique* : distance de Wasserstein entre les
distributions d'attributs cliniquement interprétables (nombre de lésions,
charge lésionnelle...) estimés sur les images réelles et générées. C'est la
seule des métriques qui soit directement lisible par un clinicien.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance

from .phantom import estimate_attributes


# --------------------------------------------------------------------------
# Outils
# --------------------------------------------------------------------------

def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xx = (X**2).sum(1)[:, None]
    yy = (Y**2).sum(1)[None, :]
    return np.maximum(xx + yy - 2.0 * X @ Y.T, 0.0)


def median_bandwidth(X: np.ndarray, max_n: int = 1000, seed: int = 0) -> float:
    """Heuristique de la médiane pour la largeur de bande du noyau RBF.

    **Elle doit être calculée une fois pour toutes sur le jeu de référence**,
    et non séparément pour chaque modèle : sinon la métrique change d'échelle
    d'un point de la courbe à l'autre et les comparaisons sont invalides.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_n, len(X)), replace=False)
    d2 = _pairwise_sq_dists(X[idx], X[idx])
    iu = np.triu_indices(len(idx), k=1)
    return float(np.sqrt(np.median(d2[iu]) / 2.0))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """Estimateur sans biais de la MMD^2 (noyau RBF, statistique en U).

    Contrairement à la FID, cet estimateur est **sans biais** : son espérance
    ne dépend pas du nombre d'échantillons. C'est décisif ici, car la courbe
    qualité/n compare des modèles évalués dans des conditions qui doivent
    rester strictement identiques.
    """
    g = 1.0 / (2.0 * bandwidth**2)
    n, m = len(X), len(Y)
    kxx = np.exp(-g * _pairwise_sq_dists(X, X))
    kyy = np.exp(-g * _pairwise_sq_dists(Y, Y))
    kxy = np.exp(-g * _pairwise_sq_dists(X, Y))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(
        kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    )


def frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Distance de Fréchet entre les approximations gaussiennes (la « FID »)."""
    mu1, mu2 = X.mean(0), Y.mean(0)
    s1, s2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def fid_infinity(
    X: np.ndarray, Y: np.ndarray, n_points: int = 12, n_reps: int = 3, seed: int = 0
) -> float:
    """FID extrapolée à un nombre infini d'échantillons (Chong & Forsyth, 2020).

    La FID est biaisée d'un terme en ~1/N. On l'estime pour plusieurs tailles
    de sous-échantillons et on extrapole la régression linéaire en 1/N vers 0.
    """
    rng = np.random.default_rng(seed)
    sizes = np.linspace(len(X) // 4, len(X), n_points).astype(int)
    inv_n, fids = [], []
    for s in sizes:
        for _ in range(n_reps):
            idx = rng.choice(len(X), size=s, replace=False)
            inv_n.append(1.0 / s)
            fids.append(frechet_distance(X[idx], Y))
    a, b = np.polyfit(inv_n, fids, 1)
    return float(b)


def _knn_radii(X: np.ndarray, k: int) -> np.ndarray:
    d2 = _pairwise_sq_dists(X, X)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])


def precision_recall_density_coverage(
    real: np.ndarray, fake: np.ndarray, k: int = 5
) -> dict[str, float]:
    """Précision/rappel (Kynkäänniemi 2019) et densité/couverture (Naeem 2020).

    ``precision`` : fraction d'échantillons générés tombant dans la variété
    estimée des données réelles (fidélité).
    ``recall`` : fraction de données réelles couvertes par la variété des
    échantillons générés (diversité).
    ``density``/``coverage`` : variantes robustes aux valeurs aberrantes.
    """
    r_real = _knn_radii(real, k)
    r_fake = _knn_radii(fake, k)
    d_rf = np.sqrt(_pairwise_sq_dists(real, fake))  # (n_real, n_fake)

    precision = float((d_rf <= r_real[:, None]).any(axis=0).mean())
    recall = float((d_rf <= r_fake[None, :]).any(axis=1).mean())
    density = float((d_rf <= r_real[:, None]).sum(axis=0).mean() / k)
    coverage = float((d_rf.min(axis=1) <= r_real).mean())
    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


def memorization(
    train: np.ndarray, fake: np.ndarray, holdout: np.ndarray
) -> dict[str, float]:
    """Audit de mémorisation, calibré par un jeu réel tenu à l'écart.

    Pour chaque image générée on mesure la distance au plus proche voisin dans
    le *jeu d'entraînement*. La même mesure est faite pour des images réelles
    tenues à l'écart : elles fournissent la distribution de référence de ce
    qu'une image « authentiquement nouvelle » doit produire.

    ``nn_ratio``
        Médiane des distances des générés / médiane pour le jeu tenu à
        l'écart. Vaut ~1 si le modèle généralise, tend vers 0 s'il recopie.
    ``authenticity``
        Fraction d'échantillons générés plus éloignés de leur plus proche
        voisin d'entraînement que ne l'est le quantile 5 % du jeu tenu à
        l'écart (adaptation de l'``AuthPct`` d'Alaa et al., 2022).
    ``copy_rate``
        Fraction d'échantillons dont la distance au plus proche voisin
        d'entraînement est inférieure à 25 % de la distance médiane
        intra-entraînement : copies quasi exactes.
    """
    d_fake = np.sqrt(_pairwise_sq_dists(fake, train)).min(axis=1)
    d_hold = np.sqrt(_pairwise_sq_dists(holdout, train)).min(axis=1)
    d_tt = np.sqrt(_pairwise_sq_dists(train, train))
    np.fill_diagonal(d_tt, np.inf)
    intra = float(np.median(d_tt.min(axis=1))) if len(train) > 1 else float("inf")

    thresh = float(np.quantile(d_hold, 0.05))
    return {
        "nn_ratio": float(np.median(d_fake) / max(np.median(d_hold), 1e-12)),
        "authenticity": float((d_fake > thresh).mean()),
        "copy_rate": float((d_fake < 0.25 * intra).mean()) if np.isfinite(intra) else 0.0,
    }


ATTRIBUTES = ("n_lesions", "lesion_area", "head_area", "lesion_ecc")


def attribute_discrepancy(real_images: np.ndarray, fake_images: np.ndarray) -> dict[str, float]:
    """Distance de Wasserstein-1 entre distributions d'attributs cliniques.

    Les attributs sont *estimés* par le même opérateur de mesure sur les images
    réelles et générées : les biais de l'estimateur s'annulent donc en grande
    partie dans la comparaison.
    """
    a_real = estimate_attributes(real_images)
    a_fake = estimate_attributes(fake_images)
    out = {}
    for k in ATTRIBUTES:
        r, f = a_real[k], a_fake[k]
        scale = np.std(r) + 1e-8
        out[f"w1_{k}"] = float(wasserstein_distance(r, f) / scale)
    out["w1_mean"] = float(np.mean([out[f"w1_{k}"] for k in ATTRIBUTES]))
    return out


@dataclass
class Evaluation:
    """Résultat complet de l'évaluation d'un modèle."""

    mmd2: float
    fid: float
    fid_inf: float
    precision: float
    recall: float
    density: float
    coverage: float
    nn_ratio: float
    authenticity: float
    copy_rate: float
    w1_n_lesions: float
    w1_lesion_area: float
    w1_head_area: float
    w1_lesion_ecc: float
    w1_mean: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate(
    fake_images: np.ndarray,
    real_ref_images: np.ndarray,
    train_images: np.ndarray,
    holdout_images: np.ndarray,
    feat_net,
    bandwidth: float,
    k: int = 5,
    with_fid_inf: bool = True,
) -> Evaluation:
    """Évalue un modèle à partir d'échantillons générés.

    Parameters
    ----------
    fake_images : images générées, ``(N_gen, 1, H, W)``.
    real_ref_images : jeu de référence réel, disjoint de l'entraînement.
    train_images : les n images ayant servi à l'adaptation (audit de mémorisation).
    holdout_images : second jeu réel, disjoint des deux précédents, qui calibre
        l'audit de mémorisation.
    feat_net : encodeur figé (cf. :mod:`fewshotgen.features`).
    bandwidth : largeur de bande RBF, calculée une seule fois sur ``real_ref_images``.
    """
    from .features import embed

    F_fake = embed(feat_net, fake_images)
    F_real = embed(feat_net, real_ref_images)
    F_train = embed(feat_net, train_images)
    F_hold = embed(feat_net, holdout_images)

    prdc = precision_recall_density_coverage(F_real, F_fake, k=k)
    mem = memorization(F_train, F_fake, F_hold)
    attrs = attribute_discrepancy(real_ref_images, fake_images)
    return Evaluation(
        mmd2=mmd2_unbiased(F_fake, F_real, bandwidth),
        fid=frechet_distance(F_fake, F_real),
        fid_inf=fid_infinity(F_fake, F_real) if with_fid_inf else float("nan"),
        **prdc,
        **mem,
        **attrs,
    )
