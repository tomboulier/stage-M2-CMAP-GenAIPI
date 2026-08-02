#!/usr/bin/env python3
"""Analyse des résultats du balayage : lois d'échelle, n*, figures.

    python scripts/analyze.py

Produit ``results/summary.json``, ``results/table.csv`` et les figures dans
``figures/``.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fewshotgen.pipeline import ExperimentConfig, load_jsonl  # noqa: E402
from fewshotgen.scaling import bootstrap_scaling, data_multiplier  # noqa: E402

RUNS = "results/runs.jsonl"
FIX = "results/fixtures.json"

ARM_LABEL = {
    "pretrained": "pré-entraîné",
    "scratch": "à partir de zéro",
    "pretrained_aug": "pré-entraîné + augmentation",
}
ARM_COLOR = {"pretrained": "#1f77b4", "scratch": "#d62728", "pretrained_aug": "#2ca02c"}
ARM_MARKER = {"pretrained": "o", "scratch": "s", "pretrained_aug": "^"}


def group(records: list[dict], arm: str, key: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for r in records:
        if r["arm"] == arm:
            out.setdefault(int(r["n"]), []).append(float(r[key]))
    return out


def _mean_sd(d: dict[int, list[float]], ns: list[int]):
    m = np.array([np.mean(d[n]) for n in ns])
    s = np.array([np.std(d[n], ddof=1) / np.sqrt(len(d[n])) if len(d[n]) > 1 else 0.0 for n in ns])
    return m, s


def zero_shot_value(records, key="mmd2"):
    """Valeur du contrôle n = 0 (modèle pré-entraîné sans adaptation)."""
    vals = [float(r[key]) for r in records if r["arm"] == "zero-shot"]
    return float(np.mean(vals)) if vals else None


def fig_learning_curves(records, arms, tau, outfile):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fits = {}
    zs = zero_shot_value(records)
    if zs is not None:
        ax.axhline(zs, color="#7f7f7f", lw=1.4, ls="--")
        ax.text(0.02, 0.965, "$n=0$ : pré-entraîné sans adaptation",
                transform=ax.transAxes, fontsize=9, color="#4d4d4d", va="top")
    for arm in arms:
        d = group(records, arm, "mmd2")
        if not d:
            continue
        ns = sorted(d)
        m, s = _mean_sd(d, ns)
        # La MMD^2 sans biais peut être négative près du plancher : on la
        # tronque pour l'affichage logarithmique, en le signalant.
        m_plot = np.maximum(m, 1e-5)
        ax.errorbar(ns, m_plot, yerr=s, fmt=ARM_MARKER[arm], color=ARM_COLOR[arm],
                    capsize=3, label=ARM_LABEL[arm], ms=6, lw=1.4)
        res = bootstrap_scaling(np.array(ns), d, tau=tau, n_boot=1000, seed=0)
        fits[arm] = res
        grid = np.logspace(np.log10(min(ns)), np.log10(max(ns) * 4), 100)
        ax.plot(grid, np.maximum(res.fit.predict(grid), 1e-5), "--",
                color=ARM_COLOR[arm], lw=1.2, alpha=0.8)
        if np.isfinite(res.fit.q_inf) and res.fit.q_inf > 1e-5:
            ax.axhline(res.fit.q_inf, color=ARM_COLOR[arm], lw=0.8, ls=":", alpha=0.6)

    ax.axhline(tau, color="k", lw=1.2, ls="-.")
    ax.text(ax.get_xlim()[0], tau * 1.15, r"$\tau_{95}$ : indistinguable du réel",
            fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nombre d'images d'entraînement $n$")
    ax.set_ylabel(r"MMD$^2$ (sans biais) au jeu réel de référence")
    ax.set_title("Courbe d'apprentissage de la qualité générative", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return fits


def fig_panels(records, arms, outfile):
    keys = [
        ("precision", "précision (fidélité)", True),
        ("coverage", "couverture (diversité)", True),
        ("authenticity", "authenticité (nouveauté)", True),
        ("copy_rate", "taux de copie", False),
        ("w1_n_lesions", r"$W_1$ nombre de lésions", False),
        ("w1_mean", r"$W_1$ moyen (attributs)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    for ax, (key, label, higher_better) in zip(axes.ravel(), keys):
        zs = zero_shot_value(records, key)
        if zs is not None:
            ax.axhline(zs, color="#7f7f7f", lw=1.2, ls="--", label="$n=0$")
        for arm in arms:
            d = group(records, arm, key)
            if not d:
                continue
            ns = sorted(d)
            m, s = _mean_sd(d, ns)
            ax.errorbar(ns, m, yerr=s, fmt=ARM_MARKER[arm], color=ARM_COLOR[arm],
                        capsize=2, ms=5, lw=1.3, label=ARM_LABEL[arm])
        ax.set_xscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(label + ("  ↑" if higher_better else "  ↓"), fontsize=10)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Décomposition de la « qualité » : un scalaire ne suffit pas", fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def fig_quality_vs_novelty(records, arms, outfile):
    """Le piège central : bonne adéquation obtenue par mémorisation."""
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for arm in arms:
        pts = [(r["mmd2"], r["authenticity"], r["n"]) for r in records if r["arm"] == arm]
        if not pts:
            continue
        q, a, n = np.array(pts).T
        sc = ax.scatter(np.maximum(q, 1e-5), a, c=np.log2(n), cmap="viridis",
                        marker=ARM_MARKER[arm], s=48, edgecolor=ARM_COLOR[arm], lw=1.2,
                        label=ARM_LABEL[arm])
    ax.set_xscale("log")
    ax.set_xlabel(r"MMD$^2$ (plus bas = meilleure adéquation) $\rightarrow$")
    ax.set_ylabel("authenticité (1 = aucune mémorisation)")
    ax.set_title("Adéquation distributionnelle contre nouveauté", fontsize=12)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_2 n$")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def fig_samples(cfg, arms, outfile, ns=(8, 32, 256), seed=0):
    from fewshotgen.phantom import sample_dataset

    rows = (
        [("réel (cible)", None, None)]
        + ([("pré-entraîné, n=0", "zero-shot", 0)]
           if os.path.exists("results/samples/zero-shot_n0_s0.npy") else [])
        + [(f"{ARM_LABEL[a]}, n={n}", a, n) for a in arms for n in ns]
    )
    fig, axes = plt.subplots(len(rows), 8, figsize=(8.6, 1.12 * len(rows)))
    real, _ = sample_dataset(8, "target", cfg.size, seed=99)
    for r, (label, arm, n) in enumerate(rows):
        if arm is None:
            imgs = real[:, 0]
        else:
            path = f"results/samples/{arm}_n{n}_s{seed}.npy"
            imgs = np.load(path)[:8, 0].astype(np.float32) if os.path.exists(path) else None
        for c in range(8):
            ax = axes[r, c]
            ax.axis("off")
            if imgs is not None and c < len(imgs):
                # Fenêtre d'affichage resserrée autour de la densité du sang
                # aigu (analogue d'une fenêtre AVC) : dans la fenêtre
                # parenchymateuse large, une hémorragie fraîche est peu
                # contrastée et l'inspection visuelle est trompeuse.
                ax.imshow(imgs[c], cmap="gray", vmin=-0.25, vmax=0.55,
                          interpolation="nearest")
        axes[r, 0].set_ylabel(label, fontsize=7)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
    fig.suptitle("Échantillons (fenêtre d'affichage resserrée sur le sang aigu)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def fig_n_star(fits, taus, outfile):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    width = 0.8 / max(len(fits), 1)
    for i, (arm, res_by_tau) in enumerate(fits.items()):
        xs, ys, los, his = [], [], [], []
        for j, (name, res) in enumerate(res_by_tau.items()):
            xs.append(j + (i - len(fits) / 2 + 0.5) * width)
            v = res.n_star
            ys.append(min(v, 1e6) if np.isfinite(v) else 1e6)
            lo, hi = res.n_star_ci
            los.append(max(lo, 1.0) if np.isfinite(lo) else 1.0)
            his.append(min(hi, 1e6) if np.isfinite(hi) else 1e6)
        ys = np.array(ys)
        ax.bar(xs, ys, width=width * 0.9, color=ARM_COLOR[arm], label=ARM_LABEL[arm], alpha=0.85)
        ax.errorbar(xs, ys, yerr=[ys - np.array(los), np.array(his) - ys],
                    fmt="none", ecolor="k", capsize=3, lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels(list(taus))
    ax.set_ylabel(r"$n^\star$ (images), échelle log")
    ax.set_xlabel("seuil de qualité visé")
    ax.set_title(r"Nombre d'images nécessaire $n^\star(\tau)$, IC bootstrap 95 %", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    records = load_jsonl(RUNS)
    if not records:
        print("aucun résultat dans", RUNS)
        return
    with open(FIX) as f:
        meta = json.load(f)
    cfg = ExperimentConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in meta["config"].items()})
    tau95 = meta["tau_null"]["tau_95"]
    arms = [a for a in cfg.arms if any(r["arm"] == a for r in records)]
    complete = {a: sorted({r["n"] for r in records if r["arm"] == a}) for a in arms}
    print("bras disponibles :", {a: len(v) for a, v in complete.items()})

    fits = fig_learning_curves(records, arms, tau95, "figures/01-courbes-apprentissage.png")
    fig_panels(records, arms, "figures/02-axes-de-qualite.png")
    fig_quality_vs_novelty(records, arms, "figures/03-adequation-vs-nouveaute.png")
    fig_samples(cfg, arms, "figures/04-echantillons.png")

    taus = {"τ₉₅": tau95, "2·τ₉₅": 2 * tau95, "5·τ₉₅": 5 * tau95, "10·τ₉₅": 10 * tau95}
    n_star_fits: dict[str, dict] = {}
    for arm in arms:
        d = group(records, arm, "mmd2")
        ns = np.array(sorted(d))
        n_star_fits[arm] = {
            name: bootstrap_scaling(ns, d, tau=t, n_boot=1000, seed=0)
            for name, t in taus.items()
        }
    fig_n_star(n_star_fits, taus, "figures/05-n-etoile.png")

    summary = {"tau_95": tau95, "null": meta["tau_null"], "arms": {}}
    for arm in arms:
        res = fits[arm]
        entry = {
            "alpha": res.fit.alpha,
            "alpha_ci": res.alpha_ci,
            "q_inf": res.fit.q_inf,
            "q_inf_ci": res.q_inf_ci,
            "residual_rms_log": res.fit.residual_rms_log,
            "n_star": {},
        }
        for name, r in n_star_fits[arm].items():
            entry["n_star"][name] = {
                "value": r.n_star if np.isfinite(r.n_star) else None,
                "ci": [None if not np.isfinite(x) else x for x in r.n_star_ci],
                "p_unreachable": r.p_unreachable,
            }
        summary["arms"][arm] = entry

    if "pretrained" in fits and "scratch" in fits:
        levels = {}
        for mult in (2, 5, 10):
            q = mult * tau95
            levels[f"{mult}·τ₉₅"] = data_multiplier(fits["pretrained"].fit,
                                                    fits["scratch"].fit, q)
        summary["data_multiplier_pretrained_vs_scratch"] = levels

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    keys = ["mmd2", "fid", "fid_inf", "precision", "recall", "density", "coverage",
            "nn_ratio", "authenticity", "copy_rate", "w1_n_lesions", "w1_mean"]
    with open("results/table.csv", "w") as f:
        f.write("arm,n,n_runs," + ",".join(f"{k}_mean,{k}_se" for k in keys) + "\n")
        for arm in arms:
            for n in sorted({r["n"] for r in records if r["arm"] == arm}):
                rs = [r for r in records if r["arm"] == arm and r["n"] == n]
                cells = []
                for k in keys:
                    v = np.array([r[k] for r in rs])
                    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                    cells += [f"{v.mean():.6g}", f"{se:.3g}"]
                f.write(f"{arm},{n},{len(rs)}," + ",".join(cells) + "\n")

    print(json.dumps(summary, indent=2, default=float)[:2500])
    print("\nfigures écrites dans figures/, tableaux dans results/")


if __name__ == "__main__":
    main()
