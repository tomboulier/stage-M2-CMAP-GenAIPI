"""Combien d'images faut-il pour adapter un modèle génératif pré-entraîné ?

Boîte à outils de l'expérience pilote associée à la proposition de stage M2
« Apprentissage de modèles génératifs d'images médicales 3D avec peu de
données » (CMAP / CHU Grenoble Alpes, projet GenAIPI-TBI).

Modules
-------
``phantom``   loi génératrice synthétique connue (coupes de scanner cérébral)
``unet``      UNet epsilon-prédictif compact
``ddpm``      diffusion : bruitage, entraînement, échantillonnage DDIM
``features``  espace de représentation spécifique au domaine
``metrics``   fidélité / diversité / adéquation / nouveauté / sémantique
``scaling``   loi d'échelle, n*(tau) et son intervalle de confiance
``pipeline``  orchestration d'une expérience complète
"""

__all__ = ["phantom", "unet", "ddpm", "features", "metrics", "scaling", "pipeline"]
__version__ = "0.1.0"
