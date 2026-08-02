"""Générateur de fantômes 2D de scanners cérébraux.

Ce module fournit un *processus génératif connu* qui imite grossièrement une
coupe axiale de scanner cérébral : boîte crânienne hyperdense, parenchyme,
ventricules hypodenses, et (dans le domaine cible) lésions hémorragiques
hyperdenses.

L'intérêt d'un fantôme plutôt que de vraies images pour l'expérience pilote :

1. la loi cible ``p*`` est connue analytiquement, donc les métriques
   d'évaluation peuvent être *validées* (on sait ce qu'elles devraient dire) ;
2. on contrôle exactement l'écart entre le domaine de pré-entraînement et le
   domaine cible, ce qui est le paramètre déterminant du transfert ;
3. aucune donnée de santé n'est nécessaire, et l'expérience est reproductible
   à partir d'une graine.

Deux domaines sont définis :

``source``
    Le domaine de « pré-entraînement » : des têtes variées, sans lésion
    hémorragique franche. Joue le rôle des ~39 000 volumes de MAISI.
``target``
    Le domaine « clinique » : têtes avec hémorragies (hématomes extra-axiaux
    périphériques et contusions parenchymateuses), déviation de la ligne
    médiane. Joue le rôle de CQ-500 / de la cohorte GenAIPI-TBI.

Les intensités sont exprimées en unités Hounsfield (UH) puis fenêtrées
(fenêtre parenchymateuse WL=40, WW=160) et ramenées dans [-1, 1], ce qui est
le prétraitement usuel avant un modèle de diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

# Unités Hounsfield approximatives des tissus d'intérêt.
HU_AIR = -1000.0
HU_BRAIN = 35.0
HU_WHITE_MATTER = 28.0
HU_CSF = 8.0
HU_BLOOD_ACUTE = 72.0
HU_BONE = 1100.0

# Fenêtre parenchymateuse : centre 40 UH, largeur 160 UH.
WINDOW_LEVEL = 40.0
WINDOW_WIDTH = 160.0


def window(hu: np.ndarray) -> np.ndarray:
    """Applique la fenêtre parenchymateuse et renvoie des valeurs dans [-1, 1]."""
    lo = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    hi = WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    x = (hu - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0) * 2.0 - 1.0


def unwindow(x: np.ndarray) -> np.ndarray:
    """Inverse de :func:`window` (utile pour interpréter les images générées)."""
    lo = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    hi = WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    return (x + 1.0) / 2.0 * (hi - lo) + lo


@dataclass
class DomainConfig:
    """Paramètres de la loi génératrice d'un domaine."""

    name: str
    # Géométrie de la tête (en fraction de la taille de l'image).
    head_radius: tuple[float, float] = (0.34, 0.42)
    head_anisotropy: tuple[float, float] = (0.80, 0.95)
    head_center_jitter: float = 0.02
    skull_thickness: tuple[float, float] = (0.030, 0.055)
    # Ventricules.
    ventricle_scale: tuple[float, float] = (0.10, 0.18)
    ventricle_offset: float = 0.10
    # Lésions hémorragiques.
    n_lesions: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.array([0, 1, 2]), np.array([0.85, 0.13, 0.02]))
    )
    lesion_radius: tuple[float, float] = (0.03, 0.06)
    lesion_hu: tuple[float, float] = (55.0, 65.0)
    # Fraction de lésions extra-axiales (collées à la table interne du crâne).
    p_extraaxial: float = 0.3
    # Déviation de la ligne médiane (effet de masse).
    midline_shift: tuple[float, float] = (0.0, 0.0)
    # Bruit et flou.
    noise_hu: float = 6.0
    blur: tuple[float, float] = (0.6, 1.0)


SOURCE_DOMAIN = DomainConfig(
    name="source",
    head_radius=(0.32, 0.44),
    head_anisotropy=(0.78, 0.98),
    skull_thickness=(0.028, 0.060),
    ventricle_scale=(0.08, 0.20),
    # Domaine « générique » : hémorragie rare et discrète.
    n_lesions=(np.array([0, 1]), np.array([0.9, 0.1])),
    lesion_radius=(0.025, 0.045),
    lesion_hu=(50.0, 58.0),
    p_extraaxial=0.15,
    midline_shift=(0.0, 0.0),
    noise_hu=6.0,
)

TARGET_DOMAIN = DomainConfig(
    name="target",
    head_radius=(0.35, 0.41),
    head_anisotropy=(0.82, 0.94),
    skull_thickness=(0.034, 0.050),
    ventricle_scale=(0.09, 0.17),
    # Cohorte traumatisme crânien : au moins une lésion dans ~85 % des cas,
    # plus volumineuses et plus denses (sang aigu).
    n_lesions=(np.array([0, 1, 2, 3]), np.array([0.15, 0.42, 0.30, 0.13])),
    lesion_radius=(0.045, 0.095),
    lesion_hu=(66.0, 80.0),
    p_extraaxial=0.55,
    midline_shift=(0.0, 0.045),
    noise_hu=7.0,
)

DOMAINS = {"source": SOURCE_DOMAIN, "target": TARGET_DOMAIN}


def _ellipse_mask(
    yy: np.ndarray,
    xx: np.ndarray,
    cy: float,
    cx: float,
    ry: float,
    rx: float,
    angle: float = 0.0,
) -> np.ndarray:
    """Distance normalisée au centre d'une ellipse orientée (<= 1 à l'intérieur)."""
    ca, sa = np.cos(angle), np.sin(angle)
    dy, dx = yy - cy, xx - cx
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    return (u / rx) ** 2 + (v / ry) ** 2


def sample_phantom(
    rng: np.random.Generator, size: int, cfg: DomainConfig
) -> tuple[np.ndarray, dict[str, float]]:
    """Tire une coupe et renvoie ``(image, attributs_latents)``.

    ``image`` est de forme ``(size, size)``, à valeurs dans [-1, 1].
    ``attributs_latents`` contient les paramètres réellement tirés : ils
    servent de vérité terrain pour l'évaluation sémantique (on ne les observe
    évidemment pas sur les images générées, où on les *estime*).
    """
    lin = (np.arange(size) + 0.5) / size
    yy, xx = np.meshgrid(lin, lin, indexing="ij")

    cy = 0.5 + rng.uniform(-cfg.head_center_jitter, cfg.head_center_jitter)
    cx = 0.5 + rng.uniform(-cfg.head_center_jitter, cfg.head_center_jitter)
    ry = rng.uniform(*cfg.head_radius)
    rx = ry * rng.uniform(*cfg.head_anisotropy)
    tilt = rng.uniform(-0.12, 0.12)

    img = np.full((size, size), HU_AIR, dtype=np.float64)

    # Crâne : couronne entre l'ellipse externe et l'ellipse interne.
    thick = rng.uniform(*cfg.skull_thickness)
    outer = _ellipse_mask(yy, xx, cy, cx, ry, rx, tilt) <= 1.0
    inner_ry, inner_rx = ry - thick, rx - thick
    inner = _ellipse_mask(yy, xx, cy, cx, inner_ry, inner_rx, tilt) <= 1.0
    img[outer] = HU_BONE * rng.uniform(0.85, 1.05)

    # Parenchyme, avec un léger gradient substance blanche / substance grise.
    brain_hu = rng.normal(HU_BRAIN, 1.5)
    depth = _ellipse_mask(yy, xx, cy, cx, inner_ry, inner_rx, tilt)
    parenchyma = brain_hu - (HU_BRAIN - HU_WHITE_MATTER) * np.clip(1.0 - depth, 0.0, 1.0)
    img[inner] = parenchyma[inner]

    # Ventricules latéraux : deux ellipses hypodenses, décalées de la ligne
    # médiane, éventuellement comprimées par un effet de masse.
    shift = rng.uniform(*cfg.midline_shift) * rng.choice([-1.0, 1.0])
    vs = rng.uniform(*cfg.ventricle_scale)
    for side in (-1.0, 1.0):
        vcx = cx + side * cfg.ventricle_offset * rx + shift
        vcy = cy + rng.uniform(-0.02, 0.02)
        vry = vs * ry * rng.uniform(0.9, 1.3)
        vrx = vs * rx * rng.uniform(0.45, 0.75)
        ven = _ellipse_mask(yy, xx, vcy, vcx, vry, vrx, tilt) <= 1.0
        img[ven & inner] = rng.normal(HU_CSF, 2.0)

    # Lésions hémorragiques.
    counts, probs = cfg.n_lesions
    n_les = int(rng.choice(counts, p=probs))
    lesion_mask = np.zeros((size, size), dtype=bool)
    lesion_hu_values: list[float] = []
    for _ in range(n_les):
        lr = rng.uniform(*cfg.lesion_radius)
        if rng.uniform() < cfg.p_extraaxial:
            # Hématome extra-axial : lentille collée à la table interne.
            theta = rng.uniform(0, 2 * np.pi)
            rad = 1.0 - lr / max(inner_ry, inner_rx) * 0.9
            ly = cy + rad * inner_ry * np.sin(theta)
            lx = cx + rad * inner_rx * np.cos(theta)
            lry, lrx = lr * rng.uniform(0.55, 0.80), lr * rng.uniform(1.3, 2.0)
            langle = theta + np.pi / 2
        else:
            # Contusion parenchymateuse : blob plus isotrope, plus profond.
            theta = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0.15, 0.75)
            ly = cy + rad * inner_ry * np.sin(theta)
            lx = cx + rad * inner_rx * np.cos(theta)
            lry, lrx = lr * rng.uniform(0.7, 1.2), lr * rng.uniform(0.7, 1.2)
            langle = rng.uniform(0, np.pi)
        blob = (_ellipse_mask(yy, xx, ly, lx, lry, lrx, langle) <= 1.0) & inner
        if blob.any():
            hu = rng.uniform(*cfg.lesion_hu)
            img[blob] = hu
            lesion_mask |= blob
            lesion_hu_values.append(hu)

    # Bruit du détecteur puis flou : la reconstruction tomographique corrèle
    # spatialement le bruit, l'ordre inverse produirait un bruit blanc
    # irréaliste (et des faux positifs de détection au seuil).
    img = img + rng.normal(0.0, cfg.noise_hu, img.shape)
    img = gaussian_filter(img, sigma=rng.uniform(*cfg.blur))

    attrs = {
        "n_lesions": float(n_les),
        "lesion_area": float(lesion_mask.mean()),
        "lesion_hu": float(np.mean(lesion_hu_values)) if lesion_hu_values else float("nan"),
        "head_area": float(outer.mean()),
        "midline_shift": float(shift),
    }
    return window(img).astype(np.float32), attrs


def sample_dataset(
    n: int, domain: str = "target", size: int = 32, seed: int = 0
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tire ``n`` coupes i.i.d. du domaine demandé.

    Renvoie ``(images, attributs)`` où ``images`` a la forme ``(n, 1, size, size)``.
    """
    cfg = DOMAINS[domain]
    rng = np.random.default_rng(seed)
    imgs = np.empty((n, 1, size, size), dtype=np.float32)
    attrs: dict[str, list[float]] = {}
    for i in range(n):
        img, a = sample_phantom(rng, size, cfg)
        imgs[i, 0] = img
        for k, v in a.items():
            attrs.setdefault(k, []).append(v)
    return imgs, {k: np.asarray(v) for k, v in attrs.items()}


# --------------------------------------------------------------------------
# Estimateurs d'attributs : applicables aux images *générées*, où les
# paramètres latents ne sont pas disponibles. Ce sont eux qui permettent
# l'évaluation « sémantique » (§ metrics.attribute_discrepancy).
# --------------------------------------------------------------------------

#: Seuil d'hyperdensité en UH : au-delà, un pixel intracrânien est compatible
#: avec du sang aigu. Volontairement conservateur (le parenchyme est à ~35 UH,
#: le sang aigu à 60-80 UH).
HYPERDENSE_HU = 52.0
#: Seuil « os » : la fenêtre parenchymateuse sature à 120 UH, l'os y est donc
#: écrêté. Un seuil à 100 UH sépare le crâne de tout tissu mou, y compris du
#: sang aigu très dense.
BONE_HU = 100.0


def estimate_attributes(images: np.ndarray) -> dict[str, np.ndarray]:
    """Estime des attributs cliniquement interprétables sur un lot d'images.

    ``images`` : ``(n, 1, H, W)`` dans [-1, 1]. Aucune information latente
    n'est utilisée, donc la fonction s'applique indifféremment à des images
    réelles ou générées.
    """
    from scipy.ndimage import binary_erosion, binary_fill_holes, label

    hu = unwindow(images[:, 0].astype(np.float64))
    n = hu.shape[0]
    out = {k: np.zeros(n) for k in ("n_lesions", "lesion_area", "head_area", "lesion_ecc")}
    for i in range(n):
        bone = hu[i] >= BONE_HU
        head = binary_fill_holes(bone)
        if head is None:
            head = bone
        # Érosion d'un pixel : supprime la couronne de volume partiel à
        # l'interface os / tissu, qui traverse mécaniquement le seuil
        # d'hyperdensité et produirait des lésions fantômes.
        intra = binary_erosion(head & ~bone, np.ones((3, 3)), border_value=0)
        lesion = intra & (hu[i] >= HYPERDENSE_HU)
        lab, k = label(lesion)
        # On ignore les composantes de 1 pixel (bruit).
        sizes = np.bincount(lab.ravel())[1:] if k else np.array([])
        keep = sizes >= 2
        out["n_lesions"][i] = float(keep.sum())
        out["lesion_area"][i] = float(lesion.sum()) / max(float(head.sum()), 1.0)
        out["head_area"][i] = float(head.mean())
        # Excentricité moyenne des lésions : distance au centre de la tête,
        # normalisée par le rayon. Distingue extra-axial (~1) de central (~0).
        if keep.any():
            ys, xs = np.nonzero(lesion)
            hy, hx = np.nonzero(head)
            cy, cx = hy.mean(), hx.mean()
            r = np.sqrt(((hy - cy) ** 2 + (hx - cx) ** 2).mean()) * np.sqrt(2.0)
            out["lesion_ecc"][i] = float(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).mean() / max(r, 1e-6))
        else:
            out["lesion_ecc"][i] = 0.0
    return out
