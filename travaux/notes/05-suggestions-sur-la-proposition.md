# Suggestions de rédaction sur la proposition de stage

Le document est clair, bien référencé et se lit vite — c'est déjà l'essentiel.
Les suggestions ci-dessous portent sur la **précision de la question posée** et
sur ce qu'un bon candidat en mathématiques appliquées cherchera dans l'annonce.
Elles sont classées par utilité décroissante ; les trois premières me semblent
valoir le coup, les suivantes sont cosmétiques.

Les corrections purement typographiques et LaTeX ont été appliquées séparément
sur `main` (accents sur les capitales, chargement de `babel-french`, référence
[2] reformatée, métadonnées PDF).

---

## 1. Préciser ce que « seuil de qualité » désigne — priorité haute

**Passage actuel** (Objectif du stage) :

> quel est le nombre minimal d'images d'entraînement nécessaire pour qu'un
> modèle génératif pré-entraîné produise, après adaptation, des images
> atteignant un seuil de qualité donné ?

Le risque est double : un candidat comprendra « seuil de FID », et le
stagiaire recruté passera trois mois à produire une courbe qui, faute de
contrôle de la mémorisation, mesurera en partie la capacité du modèle à
recopier ses données (cf. [note 01 §1](01-analyse-critique-du-sujet.md)).

**Ajout proposé**, juste après la question, sans la modifier :

> Répondre à cette question suppose de définir « qualité » de façon non
> ambiguë. Un modèle qui recopie ses images d'entraînement obtient une
> fidélité parfaite et un excellent score aux métriques usuelles, tout en
> étant sans valeur et en constituant une fuite de données de santé. Le stage
> comportera donc un volet méthodologique : construire un critère
> d'évaluation combinant adéquation distributionnelle, diversité et
> **nouveauté** vis-à-vis du jeu d'entraînement.

Cela améliore aussi l'annonce : le volet méthodologique est ce qui rend le
sujet attractif pour un M2 de maths appliquées, plus que le fine-tuning.

## 2. Faire apparaître le versant théorique — priorité haute

La référence [12] (Chen et al., ICLR 2023) est citée pour ses garanties
d'échantillonnage, mais ces garanties sont *conditionnelles* à l'erreur
d'estimation du score : elles ne disent rien du nombre d'échantillons
nécessaire, qui est précisément la question du stage.

Les deux références qui font ce lien sont **Oko, Akiyama & Suzuki (ICML 2023)**
— vitesses d'estimation de distribution quasi minimax — et **Chen, Huang, Zhao
& Wang (ICML 2023)** — vitesses gouvernées par la dimension *intrinsèque* de la
variété des données, et non par la dimension ambiante. Ce second résultat est
la justification théorique du projet et fournit une prédiction testable
(cf. [note 01 §7](01-analyse-critique-du-sujet.md#7)).

**Ajout proposé** en fin d'état de l'art :

> Sur le plan théorique, la vitesse d'estimation d'une loi par un modèle de
> diffusion est gouvernée par la dimension intrinsèque du support des données
> plutôt que par la dimension ambiante [Chen et al., ICML 2023 ; Oko et al.,
> ICML 2023]. Le stage pourra confronter l'exposant empirique de la loi
> d'échelle mesurée à la dimension intrinsèque estimée sur les mêmes données,
> ce qui relierait directement la partie expérimentale à ce cadre théorique.

C'est aussi ce qui signale au candidat que l'encadrement CMAP apporte autre
chose qu'un accès GPU.

## 3. Mentionner le contrôle `n = 0` — priorité moyenne

MAISI est pré-entraîné sur des scanners corps entier qui incluent des
acquisitions crâniennes, et CQ-500 est public depuis 2018. Une partie de la
distribution cible est donc peut-être déjà couverte avant toute adaptation. Une
phrase suffit à le signaler :

> L'étude comportera un point de référence à `n = 0` (échantillonnage du
> modèle pré-entraîné sans adaptation), afin de mesurer la part de la
> distribution cible déjà couverte par le pré-entraînement.

## 4. « Modèles génératifs » vs « modèles de diffusion »

Le titre et l'objectif disent « modèles génératifs », l'état de l'art ne parle
que de diffusion. C'est un choix défendable — et probablement le bon — mais
autant l'assumer explicitement, par exemple en fin de contexte : « Nous nous
restreignons aux modèles de diffusion, devenus l'approche de référence en
imagerie médicale [4]. »

## 5. PyTorch / TensorFlow

MAISI, MONAI et BLAST-CT sont en PyTorch. Mentionner TensorFlow dans le profil
recherché risque d'attirer des candidatures mal orientées et n'apporte rien.
Suggestion : « Compétences en programmation Python ; expérience avec PyTorch
(l'écosystème MONAI est utilisé dans le projet) ».

## 6. Détails

- « 700 patients et 1 600 scanners » : préciser si `n` sera compté en patients
  ou en examens change beaucoup la lecture. Compter en patients (les examens
  d'un même patient sont fortement dépendants).
- Clé bibliographique `chen2022sampling` pour un article ICLR 2023 : sans
  conséquence sur le rendu, mais source de confusion à la relecture.
- La référence [5] (MAISI) est parue depuis à WACV 2025 ; citer la version
  publiée plutôt que l'arXiv.
- Le stage étant au CMAP avec des données du CHU de Grenoble, préciser le
  cadre d'accès aux données (les données restent-elles à Grenoble ? le
  stagiaire y fait-il des séjours ?) éviterait des questions en entretien.
