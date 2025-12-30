# Proposition de stage M2 - CMAP École Polytechnique

## Apprentissage de modèles génératifs d'images médicales 3D avec peu de données

### Résumé

Ce dépôt contient la proposition de stage de Master 2 portant sur l'étude du fine-tuning de modèles génératifs (MAISI) pour la synthèse d'images CT cérébrales avec un nombre limité de données d'entraînement.

### Encadrants

- **Josselin Garnier** - CMAP, École Polytechnique
- **Thomas Boulier** - CHU Grenoble Alpes, PREDIMED

### Période

Avril - Septembre 2026 (6 mois)

### Utilisation

```bash
# Afficher l'aide
make

# Installer les dépendances LaTeX (Ubuntu/Debian)
make install

# Compiler le PDF
make pdf

# Compiler et ouvrir le PDF
make view

# Nettoyer les fichiers auxiliaires
make clean
```

### Structure du projet

```
.
├── proposition-stage.tex   # Document principal
├── Makefile                # Automatisation de la compilation
├── .gitignore              # Fichiers ignorés par git
└── README.md               # Ce fichier
```

### Liens utiles

- [MAISI - NVIDIA](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/)
- [CQ-500 Dataset](http://headctstudy.qure.ai/dataset)
- [MONAI](https://monai.io/)

### Contact

- josselin.garnier@polytechnique.edu
- tboulier@chu-grenoble.fr