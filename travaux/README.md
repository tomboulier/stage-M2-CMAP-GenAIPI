# Combien d'images faut-il pour adapter un modèle génératif pré-entraîné ?

Travail exploratoire sur le sujet de stage M2
[« Apprentissage de modèles génératifs d'images médicales 3D avec peu de
données »](../proposition-stage.tex) (CMAP / CHU Grenoble Alpes, projet
GenAIPI-TBI).

Ce dossier contient trois choses :

1. une **analyse critique du sujet**, qui identifie deux difficultés
   méthodologiques suffisantes pour invalider l'étude si elles ne sont pas
   traitées dès le protocole, et propose un apport théorique ;
2. un **protocole complet et pré-enregistrable** pour l'étude sur CQ-500 et
   MAISI ;
3. une **expérience pilote réellement exécutée**, qui met en œuvre la
   méthodologie de bout en bout et produit une estimation de `n*` avec son
   intervalle de confiance.

## Documents

| | |
|---|---|
| [`notes/01-analyse-critique-du-sujet.md`](notes/01-analyse-critique-du-sujet.md) | ce qui ne marchera pas tel quel, et pourquoi |
| [`notes/02-protocole.md`](notes/02-protocole.md) | protocole applicable à CQ-500 + MAISI |
| [`notes/03-resultats-experience-pilote.md`](notes/03-resultats-experience-pilote.md) | résultats chiffrés de l'expérience pilote |
| [`notes/04-feuille-de-route.md`](notes/04-feuille-de-route.md) | découpage en 6 mois, jalons et risques |

## L'idée en une figure

![courbes d'apprentissage](figures/01-courbes-apprentissage.png)

Divergence au jeu réel en fonction du nombre d'images d'adaptation, pour un
modèle pré-entraîné et pour le même modèle appris à partir de zéro, à budget de
calcul identique. Les pointillés sont l'ajustement `Q(n) = Q∞ + a·n^(−α)` ; la
ligne horizontale est le seuil `τ₉₅` en deçà duquel les images générées ne sont
plus distinguables d'images réelles par le test utilisé. L'écart **horizontal**
entre les courbes est la quantité utile : le facteur d'économie de patients.

## Le point de méthode le plus important

Un générateur qui recopie ses `n` images d'entraînement obtient une fidélité
parfaite, une FID excellente, et n'a **aucune** valeur — tout en constituant une
fuite de données de santé. Une courbe « qualité en fonction de `n` » qui ne
contrôle pas la mémorisation mesure donc en partie la capacité à mémoriser, et
ce biais joue exactement dans le sens qui fait conclure à tort que peu de
données suffisent.

L'évaluation est donc **vectorielle** — adéquation, fidélité, diversité,
**nouveauté** — et l'axe de nouveauté est calibré sur un jeu réel tenu à
l'écart. C'est vérifié par un test unitaire :
`tests/test_metrics.py::TestMemorization::test_copies_can_have_excellent_mmd`.

## Expérience pilote

Ne disposant ni de CQ-500 ni de GPU dans cet environnement, la méthodologie est
démontrée sur un **jeu de fantômes 2D de scanners cérébraux à loi génératrice
connue** (`src/fewshotgen/phantom.py`) : boîte crânienne, parenchyme,
ventricules, hémorragies extra-axiales et contusions. Deux domaines — un
domaine « source » sans hémorragie franche qui joue le rôle des 39 000 volumes
de pré-entraînement de MAISI, un domaine « cible » avec hémorragies qui joue le
rôle de CQ-500.

Ce choix n'est pas seulement un pis-aller : il permet de **valider les
métriques**, puisqu'on connaît la loi cible et qu'on sait donc ce que chaque
métrique devrait dire.

Ce que le pilote démontre : la chaîne complète (loi d'échelle, `τ` calibré sur
la loi nulle, `n*` avec IC bootstrap, audit de mémorisation, évaluation
sémantique) est opérationnelle et donne des réponses cohérentes. Ce qu'il ne
démontre pas : les valeurs numériques de `α`, `Q∞` et `n*` obtenues ici sont
propres au fantôme et **ne se transportent pas** à CQ-500.

## Reproduire

```bash
pip install -r requirements.txt

# 1. encodeur de référence, seuil tau, pré-entraînement sur le domaine source
python scripts/run_pilot.py --prepare --threads 4

# 2. balayage (n, graine, bras). Les petits tenseurs passent mal à l'échelle
#    en multi-thread : 4 processus mono-thread valent ~2x un processus à
#    4 threads sur cette machine.
for i in 0 1 2 3; do
    OMP_NUM_THREADS=1 python scripts/run_pilot.py --shard $i --n-shards 4 &
done; wait

# 3. lois d'échelle, n*, figures, tableaux
python scripts/analyze.py

# tests (aucun entraînement lourd, quelques minutes)
python -m pytest tests/ -v
```

Les résultats sont ajoutés au fil de l'eau dans `results/runs.jsonl` : une
interruption ne perd que le run en cours et une relance reprend où l'exécution
s'est arrêtée.

## Organisation du code

```
src/fewshotgen/
  phantom.py    loi génératrice connue + estimateurs d'attributs cliniques
  unet.py       UNet epsilon-prédictif compact (~0,4 M paramètres)
  ddpm.py       planning cosinus, entraînement, échantillonnage DDIM, EMA
  features.py   espace de représentation spécifique au domaine
  metrics.py    adéquation / fidélité / diversité / nouveauté / sémantique
  scaling.py    ajustement de la loi d'échelle, n*(tau), bootstrap
  pipeline.py   plan d'expérience et orchestration
scripts/        run_pilot.py (balayage), analyze.py (figures et tableaux)
tests/          validation des métriques, de l'estimation de n*, de la diffusion
```

Le passage à CQ-500 + MAISI ne touche que trois points : le chargement des
données, l'extracteur de caractéristiques (encodeur BLAST-CT à la place du CNN
de fantômes) et la boucle d'adaptation. Le plan d'expérience, les métriques et
l'inférence statistique sont inchangés.
