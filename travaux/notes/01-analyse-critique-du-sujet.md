# Analyse critique du sujet

> « Quel est le nombre minimal d'images d'entraînement nécessaire pour qu'un
> modèle génératif pré-entraîné produise, après adaptation, des images
> atteignant un seuil de qualité donné ? »

La question est bien posée sur le plan scientifique et sans réponse établie
dans la littérature — c'est ce qui en fait un bon sujet. Elle comporte
cependant plusieurs difficultés qu'il vaut mieux identifier avant de lancer le
premier entraînement qu'après trois mois de balayage : deux d'entre elles
suffisent à invalider entièrement les conclusions si elles ne sont pas traitées
dès le protocole.

Les sections 1 et 2 sont, à mon sens, celles qui doivent conditionner la
rédaction du protocole. Les sections 3 à 6 sont des risques identifiés. La
section 7 propose un apport théorique qui me paraît être le vrai potentiel de
publication du stage.

---

## 1. « Un seuil de qualité » n'est pas un scalaire, et le scalaire usuel est piégeux

Considérons le générateur suivant : il tire uniformément l'une des `n` images
d'entraînement et la renvoie telle quelle. Ce générateur

- a une **fidélité parfaite** : toutes ses sorties sont de vraies images ;
- a une **FID qui tend vers 0** quand `n` grandit, puisque sa loi est la mesure
  empirique, qui converge vers `p*` ;
- a une **valeur d'usage nulle** : il n'apporte aucune image nouvelle ;
- est un **incident de sécurité des données** : il exfiltre des scanners de
  patients identifiables.

Il obtiendrait donc un excellent score à la métrique standard. Ce n'est pas une
construction théorique : Carlini et al. (2023) extraient des images
d'entraînement de Stable Diffusion, et Dar et al. (2023, 2024) montrent une
mémorisation massive de modèles de diffusion entraînés sur des IRM et des
scanners — précisément dans le régime de données qui nous intéresse, puisque la
mémorisation croît quand `n` diminue.

**Conséquence directe pour le sujet.** Une courbe « qualité en fonction de `n` »
mesurée à la seule FID mesure, en partie, la capacité du modèle à mémoriser.
Comme la mémorisation est *plus forte* à petit `n`, ce biais joue exactement
dans le sens qui fait conclure à tort que « peu de données suffisent ». Le
seuil de qualité doit donc être **vectoriel**, avec au minimum quatre axes :

| axe | question | métriques |
|---|---|---|
| adéquation | la loi générée est-elle celle des données ? | MMD² sans biais, KID, FID∞ |
| fidélité | les échantillons sont-ils plausibles ? | précision, densité |
| diversité | couvrent-ils tout le support ? | rappel, couverture |
| **nouveauté** | **diffèrent-ils du jeu d'entraînement ?** | AuthPct, ratio de distance au plus proche voisin, taux de copie |

L'axe de nouveauté doit être **calibré par un jeu réel tenu à l'écart** : la
bonne question n'est pas « les images générées sont-elles loin du jeu
d'entraînement ? » (loin dans quelle unité ?) mais « en sont-elles aussi loin
que de vraies images qui n'ont pas servi à l'entraînement ? ».

C'est implémenté dans `src/fewshotgen/metrics.py` et testé dans
`tests/test_metrics.py::TestMemorization::test_copies_can_have_excellent_mmd`,
qui vérifie qu'un générateur-copieur obtient bien un score d'adéquation
excellent et est bien démasqué par l'audit.

## 2. La FID est un estimateur biaisé, et le biais dépend du plan d'expérience

La FID compare deux gaussiennes ajustées sur des activations. L'estimateur
plug-in de la distance de Fréchet est **biaisé**, d'un terme en `O(1/N)` où `N`
est le nombre d'échantillons, et le biais est **positif** : entre deux
échantillons de la *même* loi, la FID mesurée n'est pas nulle et décroît quand
on augmente `N` (Chong & Forsyth, 2020). `tests/test_metrics.py` en fait un test
unitaire.

Dans un balayage sur `n`, ce biais est bénin *si et seulement si* le protocole
d'évaluation est parfaitement gelé : même nombre d'images générées, même jeu de
référence, même extracteur de caractéristiques, pour tous les points de la
courbe. Deux pratiques courantes le violent :

- générer « autant d'images qu'il y a de données d'entraînement » ;
- comparer au jeu d'entraînement plutôt qu'à un jeu de référence disjoint.

Deux corrections sont retenues ici : utiliser un estimateur **sans biais** de la
MMD² (statistique en U), dont l'espérance ne dépend pas de `N` (testé), et
reporter en parallèle la FID brute et son extrapolation FID∞ pour documenter
l'écart.

**Choix de l'espace de représentation.** InceptionV3 est entraîné sur ImageNet.
Sur du scanner cérébral, ses activations sont peu sensibles à ce qui porte le
sens clinique : une hémorragie de 20 mL ne déplace presque pas la FID ImageNet,
alors qu'un changement de noyau de reconstruction la déplace beaucoup. La
première recommandation méthodologique de ce travail est donc de calculer les
métriques dans un **espace de représentation spécifique au domaine**. Le projet
dispose du candidat idéal, déjà cité dans la proposition : l'encodeur de
**BLAST-CT**, entraîné à segmenter les lésions traumatiques. Utiliser cet
encodeur rend la métrique sensible, par construction, à ce que le projet veut
préserver. (Dans l'expérience pilote, le rôle est tenu par un petit CNN entraîné
à régresser des attributs anatomiques sur un vivier disjoint.)

**Et surtout : une évaluation sémantique.** À côté des métriques
distributionnelles, il faut comparer les distributions de grandeurs *lisibles*
— nombre de lésions, volume lésionnel, localisation, présence d'une déviation
de la ligne médiane — estimées par le *même* opérateur de mesure (BLAST-CT) sur
les images réelles et générées. Un encadrant clinicien ne peut rien faire d'une
FID ; il peut faire quelque chose d'un « le modèle sous-produit les hématomes
extra-duraux d'un facteur trois ».

## 3. « n* » est un estimand statistique, pas un point de lecture sur un graphe

Poser `Q(n) = Q∞ + a·n^(−α)` transforme la question en un problème
d'estimation, avec trois bénéfices :

1. on obtient un **intervalle de confiance** sur `n*(τ)`, seul chiffre
   décisionnel utile (« il faut entre 120 et 900 patients » est une réponse ;
   « il faut 300 patients » n'en est pas une) ;
2. on estime **`Q∞`, l'erreur irréductible** : le biais d'architecture, de
   méthode d'adaptation et de budget de calcul, que la donnée supplémentaire ne
   réduit pas ;
3. si `τ ≤ Q∞`, alors `n* = +∞`, et la réponse à la question du projet n'est
   pas un nombre de patients mais **« aucune quantité de données accessible ne
   suffira, il faut changer de méthode d'adaptation »**.

Ce troisième cas est le plus important en pratique et il est indétectable sans
modéliser le plancher : une étude qui ajuste une droite en log-log conclura
toujours à un `n*` fini, quitte à extrapoler une pente qui n'existe pas.
C'est aussi celui qui a le plus de valeur pour GenAIPI-TBI : savoir en juin
qu'il faut changer de stratégie d'adaptation vaut mieux que de découvrir en
septembre que la cohorte est trop petite.

L'inférence doit reposer sur des répétitions qui rééchantillonnent **à la fois**
la graine d'entraînement et le sous-échantillon de `n` images ; en régime de
petit `n`, la seconde source domine largement. Les sous-échantillons doivent
être tirés **indépendamment** d'un point à l'autre de la courbe : des
sous-échantillons emboîtés (les 8 contenant les 4) corrèlent les points et font
sous-estimer les intervalles.

## 4. Le contrôle qui manque au protocole proposé : `n = 0`

La proposition compare implicitement « modèle pré-entraîné adapté à `n`
images » à… rien. Il manque deux contrôles, tous deux peu coûteux :

- **`n = 0`, l'échantillonnage zero-shot.** MAISI est entraîné sur ~39 000
  volumes de scanners corps entier, qui incluent des acquisitions crâniennes.
  Une partie de la distribution CQ-500 est donc peut-être déjà couverte. Si
  `Q(0)` est déjà bas, l'essentiel de la courbe mesure la proximité de CQ-500 au
  corpus de pré-entraînement, pas l'efficacité de l'adaptation ;
- **l'apprentissage à partir de zéro**, à budget d'adaptation identique. C'est
  ce qui permet de convertir l'écart en la seule quantité interprétable pour le
  projet : le **facteur d'économie de données**, `n_zéro(q) / n_pré-entraîné(q)`
  — « combien de fois moins de patients faut-il inclure ». Un écart vertical de
  MMD ne se négocie pas avec un comité d'éthique ; un facteur 10 sur le nombre
  d'inclusions, si.

**Risque de contamination.** CQ-500 est public depuis 2018. Il n'est pas exclu
qu'il figure, en tout ou partie, dans le corpus de pré-entraînement de MAISI ou
dans celui des modèles auxquels on le comparera. Dans ce cas `n*` serait
optimiste, et non transférable à la cohorte grenobloise. Deux garde-fous :
(i) documenter explicitement le corpus de MAISI et le déclarer comme limite ;
(ii) **répliquer la mesure sur la cohorte GenAIPI-TBI**, qui n'est dans aucun
corpus public. Un écart important entre les deux `n*` est en soi un résultat
publiable sur la contamination des jeux d'évaluation en imagerie médicale.

## 5. Points de plan d'expérience à fixer avant de commencer

**Ce que « `n` » désigne.** CQ-500 compte 491 examens, mais ~10⁵ coupes. Un
modèle 2D par coupe voit beaucoup d'exemples, fortement dépendants ; la taille
d'échantillon effective est de l'ordre du nombre de patients, pas de coupes.
`n` doit être compté **au niveau patient**, et tous les découpages doivent être
faits par patient. Pour GenAIPI-TBI (700 patients, 1 600 scanners), c'est
impératif : plusieurs scanners du même patient sont quasi identiques.

**Le budget de sous-échantillonnage.** Avec 491 examens, un balayage jusqu'à
`n = 256` plus un jeu de référence et un jeu de calibration disjoints épuise le
jeu de données : à `n = 256`, il ne reste que 235 examens pour tout le reste, et
les répétitions ne sont plus indépendantes. Ce n'est pas rédhibitoire, mais cela
doit être **budgété au départ** : je proposerais de réserver 150 examens
(référence + calibration, gelés) et de tirer les sous-échantillons dans les 341
restants, en acceptant que les points `n ≥ 128` soient corrélés et en le
reflétant dans le bootstrap. Découvrir ce problème en mai coûte un mois.

**Calcul constant.** Le nombre de pas d'optimisation ne doit pas dépendre de
`n`, sinon la courbe mélange « plus de données » et « plus de calcul ». Ce point
est systématiquement mal contrôlé dans la littérature few-shot.

**Prétraitement gelé.** Épaisseur de coupe (0,625 à 5 mm dans CQ-500),
rééchantillonnage, fenêtrage, recalage : ces choix déplacent la FID bien plus
que la plupart des variantes de méthode. Ils doivent être figés avant le premier
balayage et versionnés.

## 6. Non-conditionnel aujourd'hui, conditionnel demain

L'objectif à long terme de GenAIPI-TBI est de **prédire l'évolution** des
lésions : c'est un modèle conditionnel `p(x_{t+1} | x_t)`, pas un modèle de
génération libre. Or la complexité en échantillons des deux problèmes n'est pas
la même : pour le conditionnel, ce qui compte est le nombre de *paires* et la
richesse de la loi conditionnelle, et un modèle peut très bien apprendre la loi
marginale des scanners avec 200 patients sans rien apprendre de la dynamique
lésionnelle. Le stage tel qu'il est écrit répond donc à une question
*nécessaire mais non suffisante*.

Cela ne remet pas en cause le sujet — commencer par le non-conditionnel est le
bon ordre — mais suggère de réserver le dernier mois à un **proxy conditionnel**
(par exemple : génération conditionnée par un masque de lésion, avec la même
méthodologie de courbe `n*`). Cela transforme le stage d'une étude de faisabilité
en une réponse directe à la question du projet, et c'est ce qui rendra le
mémoire publiable.

## 7. L'apport théorique qui me paraît le plus prometteur

La proposition cite Chen et al. (ICLR 2023), qui bornent l'erreur en variation
totale d'un modèle de diffusion en fonction de l'erreur d'estimation du score.
Ce résultat est *conditionnel* à l'erreur de score : il ne dit rien du nombre
d'échantillons nécessaire. La brique manquante est fournie par les travaux plus
récents d'**Oko, Akiyama & Suzuki (ICML 2023)** — les modèles de diffusion
atteignent des vitesses d'estimation de distribution quasi minimax pour des
densités de Besov — et de **Chen et al. (ICML 2023, « score approximation,
estimation and distribution recovery on low-dimensional data »)**, qui montrent
que la vitesse dépend de la **dimension intrinsèque de la variété** des données,
et non de la dimension ambiante.

C'est, à mon sens, le résultat théorique central pour ce projet, parce qu'il en
donne la justification et qu'il fournit une **prédiction falsifiable** :

> Si la vitesse est gouvernée par la dimension intrinsèque `d_int` du support,
> alors l'exposant `α` de la loi d'échelle empirique doit varier comme
> `2s / (2s + d_int)` et non avec la dimension ambiante (10⁷ voxels).
> Deux corollaires testables : (i) travailler dans l'espace latent de
> l'auto-encodeur de MAISI, qui réduit `d_int`, doit *augmenter* `α` ;
> (ii) `α` estimé sur CQ-500 doit être cohérent avec `d_int` estimé
> indépendamment sur les mêmes données (Levina–Bickel, TwoNN).

Ce programme est réalisable dans le temps du stage — l'estimation de dimension
intrinsèque est peu coûteuse une fois les latents calculés — il relie
directement la partie empirique à la partie théorique, et il me semble être ce
qui distinguerait un mémoire de M2 correct d'un article. C'est aussi le point où
l'encadrement du CMAP a le plus de valeur ajoutée.

---

## Ce qui est démontré expérimentalement dans ce dépôt

Les points 1, 2, 3 et 4 ne sont pas seulement discutés : ils sont mis en œuvre
et vérifiés sur une expérience pilote complète, décrite dans
[`03-resultats-experience-pilote.md`](03-resultats-experience-pilote.md), qui
exécute la méthodologie de bout en bout sur un jeu de fantômes de scanners
cérébraux à loi génératrice connue. Le protocole applicable à CQ-500 et à MAISI
est dans [`02-protocole.md`](02-protocole.md).

## Références au-delà de celles du sujet

- N. Carlini et al., *Extracting Training Data from Diffusion Models*, USENIX Security, 2023.
- S. U. Dar et al., *Investigating data memorization in 3D latent diffusion models for medical image synthesis*, MICCAI DGM4MICCAI, 2023.
- S. U. Dar et al., *Unconditional latent diffusion models memorize patient imaging data*, 2024 (arXiv:2402.01054).
- M. F. Chong, D. Forsyth, *Effectively Unbiased FID and Inception Score and where to find them*, CVPR, 2020.
- T. Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models*, NeurIPS, 2019.
- M. F. Naeem et al., *Reliable Fidelity and Diversity Metrics for Generative Models*, ICML, 2020.
- A. Alaa et al., *How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models*, ICML, 2022.
- K. Oko, S. Akiyama, T. Suzuki, *Diffusion Models are Minimax Optimal Distribution Estimators*, ICML, 2023.
- M. Chen, K. Huang, T. Zhao, M. Wang, *Score Approximation, Estimation and Distribution Recovery of Diffusion Models on Low-Dimensional Data*, ICML, 2023.
- E. Levina, P. Bickel, *Maximum Likelihood Estimation of Intrinsic Dimension*, NeurIPS, 2004.
