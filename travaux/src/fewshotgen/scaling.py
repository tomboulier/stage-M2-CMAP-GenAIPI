"""Estimation de la courbe d'apprentissage et du nombre d'images nécessaire.

C'est la traduction statistique de la question posée par le sujet : « quel est
le nombre minimal d'images d'entraînement pour qu'un modèle génératif
pré-entraîné produise, après adaptation, des images atteignant un seuil de
qualité donné ? »

Formalisation retenue. Soit :math:`\\hat p_n` le modèle obtenu en adaptant le
modèle pré-entraîné à :math:`n` images tirées i.i.d. de la loi cible
:math:`p^\\star`, et :math:`Q` une divergence (ici la MMD^2 sans biais dans
l'espace de représentation). On postule la loi d'échelle

.. math::  \\mathbb{E}[Q(\\hat p_n, p^\\star)] = Q_\\infty + a\\, n^{-\\alpha},

où :math:`Q_\\infty \\ge 0` est l'erreur **irréductible** : biais de
l'architecture, de la méthode d'adaptation et du budget de calcul, que la
donnée supplémentaire ne réduit pas. L'estimand est alors

.. math::  n^\\star(\\tau) = \\min\\{n : Q_\\infty + a n^{-\\alpha} \\le \\tau\\}
           = \\Big(\\tfrac{a}{\\tau - Q_\\infty}\\Big)^{1/\\alpha}.

Deux conséquences, importantes pour l'interprétation :

1. si :math:`\\tau \\le Q_\\infty`, alors :math:`n^\\star = +\\infty` : aucune
   quantité de données n'atteint le seuil, et la réponse utile n'est pas un
   nombre de patients mais « changez de méthode d'adaptation ». Une étude qui
   n'estime pas :math:`Q_\\infty` ne peut pas distinguer ce cas d'un besoin de
   données très grand ;
2. :math:`n^\\star` dépend de :math:`\\tau`, qui doit être fixé *a priori* et
   de façon interprétable. On propose ici de le calibrer sur le plancher
   réel/réel : :math:`\\tau = \\kappa\\, Q_{\\text{réel/réel}}`, où
   :math:`Q_{\\text{réel/réel}}` est la divergence mesurée entre deux
   échantillons disjoints de vraies images -- exactement la logique d'une
   marge de non-infériorité en essai clinique.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def power_law(n: np.ndarray, q_inf: float, a: float, alpha: float) -> np.ndarray:
    return q_inf + a * np.asarray(n, dtype=float) ** (-alpha)


@dataclass
class ScalingFit:
    q_inf: float
    a: float
    alpha: float
    residual_rms_log: float
    n_points: int

    def predict(self, n) -> np.ndarray:
        return power_law(np.asarray(n, dtype=float), self.q_inf, self.a, self.alpha)

    def n_star(self, tau: float) -> float:
        """Nombre d'images pour atteindre le seuil ``tau`` (``inf`` si inatteignable)."""
        if tau <= self.q_inf:
            return float("inf")
        return float((self.a / (tau - self.q_inf)) ** (1.0 / self.alpha))


def fit_power_law(
    n: np.ndarray, q: np.ndarray, allow_floor: bool = True
) -> ScalingFit:
    """Ajuste ``q = q_inf + a n^{-alpha}`` par moindres carrés sur l'échelle log.

    L'ajustement se fait sur ``log q`` : les erreurs sur une divergence sont
    multiplicatives plutôt qu'additives, et les petites valeurs (grand ``n``),
    qui portent l'information sur ``q_inf``, seraient sinon écrasées.
    """
    n = np.asarray(n, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any(q <= 0):
        shift = 1e-12
        q = np.maximum(q, shift)
    log_q = np.log(q)

    def residuals(theta):
        q_inf = np.exp(theta[0]) if allow_floor else 0.0
        a, alpha = np.exp(theta[1]), np.exp(theta[2])
        return np.log(np.maximum(q_inf + a * n ** (-alpha), 1e-300)) - log_q

    # Initialisation : régression log-log sans plancher.
    slope, intercept = np.polyfit(np.log(n), log_q, 1)
    theta0 = np.array(
        [np.log(max(q.min() * 0.5, 1e-9)), intercept, np.log(max(-slope, 1e-3))]
    )
    lo = np.array([np.log(1e-12), np.log(1e-9), np.log(1e-3)])
    hi = np.array([np.log(max(q.max(), 1e-9)), np.log(1e9), np.log(5.0)])
    theta0 = np.clip(theta0, lo + 1e-6, hi - 1e-6)
    sol = least_squares(residuals, theta0, bounds=(lo, hi), max_nfev=20000)
    q_inf = float(np.exp(sol.x[0])) if allow_floor else 0.0
    return ScalingFit(
        q_inf=q_inf,
        a=float(np.exp(sol.x[1])),
        alpha=float(np.exp(sol.x[2])),
        residual_rms_log=float(np.sqrt(np.mean(sol.fun**2))),
        n_points=len(n),
    )


@dataclass
class ScalingResult:
    fit: ScalingFit
    alpha_ci: tuple[float, float]
    q_inf_ci: tuple[float, float]
    n_star: float
    n_star_ci: tuple[float, float]
    p_unreachable: float
    tau: float


def bootstrap_scaling(
    n_values: np.ndarray,
    q_by_run: dict[int, list[float]],
    tau: float,
    n_boot: int = 2000,
    seed: int = 0,
    allow_floor: bool = True,
) -> ScalingResult:
    """Ajuste la loi d'échelle et quantifie l'incertitude par bootstrap.

    Parameters
    ----------
    n_values : tailles d'échantillon testées.
    q_by_run : pour chaque ``n``, la liste des valeurs de la divergence
        obtenues sur les répétitions (graines / sous-échantillons différents).
    tau : seuil de qualité visé.

    Le bootstrap est **stratifié par ``n``** et rééchantillonne les
    répétitions : il propage à la fois la variabilité d'entraînement (graine)
    et celle du tirage du sous-échantillon d'apprentissage, qui sont les deux
    sources dominantes en régime de petit ``n``.
    """
    n_values = np.asarray(n_values, dtype=float)
    means = np.array([np.mean(q_by_run[int(n)]) for n in n_values])
    point = fit_power_law(n_values, means, allow_floor=allow_floor)

    rng = np.random.default_rng(seed)
    alphas, floors, n_stars = [], [], []
    for _ in range(n_boot):
        resampled = []
        for n in n_values:
            vals = np.asarray(q_by_run[int(n)], dtype=float)
            resampled.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        try:
            f = fit_power_law(n_values, np.asarray(resampled), allow_floor=allow_floor)
        except Exception:
            continue
        alphas.append(f.alpha)
        floors.append(f.q_inf)
        n_stars.append(f.n_star(tau))

    alphas = np.asarray(alphas)
    floors = np.asarray(floors)
    n_stars = np.asarray(n_stars)
    finite = np.isfinite(n_stars)
    p_unreachable = float(1.0 - finite.mean()) if len(n_stars) else float("nan")
    if finite.sum() >= 10:
        lo, hi = np.percentile(n_stars[finite], [2.5, 97.5])
    else:
        lo, hi = float("nan"), float("inf")
    return ScalingResult(
        fit=point,
        alpha_ci=(float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))),
        q_inf_ci=(float(np.percentile(floors, 2.5)), float(np.percentile(floors, 97.5))),
        n_star=point.n_star(tau),
        n_star_ci=(float(lo), float(hi)),
        p_unreachable=p_unreachable,
        tau=tau,
    )


def data_multiplier(fit_a: ScalingFit, fit_b: ScalingFit, q_level: float) -> float:
    """Facteur d'économie de données de ``fit_a`` par rapport à ``fit_b``.

    Rapport ``n_b(q) / n_a(q)`` : « combien de fois moins d'images il faut à la
    méthode A pour atteindre la même qualité que B ». C'est la quantité que
    l'on souhaite réellement rapporter lorsqu'on compare pré-entraînement et
    apprentissage à partir de zéro : un écart vertical de MMD n'est pas
    interprétable, un facteur sur le nombre de patients l'est.
    """
    na, nb = fit_a.n_star(q_level), fit_b.n_star(q_level)
    if not np.isfinite(nb):
        return float("inf")
    if not np.isfinite(na) or na <= 0:
        return 0.0
    return float(nb / na)
