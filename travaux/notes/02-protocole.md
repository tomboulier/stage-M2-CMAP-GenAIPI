# Protocole expérimental — étude `n*` sur CQ-500 et MAISI

Protocole complet, prêt à pré-enregistrer, pour l'étude proposée dans le sujet.
Il applique les correctifs identifiés dans
[`01-analyse-critique-du-sujet.md`](01-analyse-critique-du-sujet.md). Le code de
`src/fewshotgen/` en est l'implémentation à l'échelle réduite : passer à
CQ-500 + MAISI consiste à remplacer trois fonctions (chargement des données,
extracteur de caractéristiques, boucle d'adaptation), pas à réécrire l'étude.

---

## 1. Question et estimand

Soit `p*` la loi des volumes de scanner cérébral de la population cible et
`p̂_n` le modèle obtenu en adaptant un modèle pré-entraîné à `n` examens tirés
i.i.d. de `p*`. Pour une divergence `Q` et un seuil `τ` :

```
n*(τ) = min { n : E[Q(p̂_n, p*)] ≤ τ }
```

sous le modèle de courbe d'apprentissage `E[Q(p̂_n, p*)] = Q∞ + a·n^(−α)`, d'où

```
n*(τ) = ( a / (τ − Q∞) )^(1/α),   et n*(τ) = +∞ si τ ≤ Q∞.
```

**Estimands rapportés** : `α`, `Q∞`, `n*(τ)` avec IC bootstrap 95 %, la
probabilité que le seuil soit hors d'atteinte, et le facteur d'économie de
données pré-entraîné/zéro.

## 2. Choix du seuil `τ` — le point à ne pas laisser arbitraire

`τ` est calibré sur la **loi nulle réel/réel** : on tire de façon répétée deux
échantillons *réels* disjoints, aux effectifs exacts de l'évaluation
(`N_gen` générées contre `N_ref` réelles), et on prend

```
τ₉₅ = quantile 95 % de la loi de Q(réel, réel).
```

Atteindre `Q ≤ τ₉₅` signifie « le jeu généré n'est pas distinguable d'un jeu
réel par ce test, à ces effectifs ». C'est l'exacte transposition d'une marge de
non-infériorité, et cela rend `n*` interprétable au lieu d'arbitraire. On
rapporte aussi `n*` pour `2τ₉₅`, `5τ₉₅` et `10τ₉₅` : la sensibilité de `n*` au
seuil fait partie du résultat, elle ne doit pas être cachée par un choix unique.

## 3. Données et découpages

**CQ-500** : 491 examens, dont 205 avec hémorragie intracrânienne.

| ensemble | taille visée | usage | gelé ? |
|---|---|---|---|
| référence `R` | 100 examens | jeu réel de comparaison pour toutes les métriques | oui |
| calibration `H` | 50 examens | calibre l'audit de nouveauté et la loi nulle | oui |
| vivier `P` | 341 examens | source des sous-échantillons d'entraînement | oui |

Découpage **stratifié sur la présence d'hémorragie** et **par patient**.
Les trois ensembles sont tirés une fois, avec une graine versionnée, et ne
changent plus. Toute modification postérieure du découpage invalide la courbe
et doit être documentée comme telle.

> **Limite à assumer d'emblée.** À `n = 256`, deux répétitions partagent en
> moyenne 75 % de leurs images : les points `n ≥ 128` sont corrélés et leurs
> intervalles de confiance sont optimistes. Trois réponses, à combiner :
> (i) borner le balayage à `n = 128` sur CQ-500 et prolonger la courbe sur la
> cohorte grenobloise (700 patients) ; (ii) rapporter l'IC sous les deux
> hypothèses (répétitions indépendantes / corrélées) ; (iii) utiliser un
> bootstrap par patient plutôt que par répétition.

**Prétraitement, figé avant le premier entraînement** : rééchantillonnage
isotrope à 1 mm³, recalage rigide sur un atlas crânien, extraction cérébrale,
fenêtrage parenchymateux (WL 40 / WW 160), normalisation dans [−1, 1], volume
recadré à une taille fixe. Version enregistrée avec les résultats.

## 4. Grille expérimentale

- `n ∈ {0, 4, 8, 16, 32, 64, 128, 256}` — **`n = 0` est le contrôle zero-shot**,
  il n'est pas optionnel ;
- `S = 5` répétitions par point (graine d'entraînement *et* sous-échantillon
  retirés indépendamment) ;
- bras d'adaptation :

| bras | description | ce qu'il teste |
|---|---|---|
| `zero-shot` | MAISI tel quel | la cible est-elle déjà couverte par le pré-entraînement ? |
| `full-ft` | tous les poids | référence haute, sur-apprentissage attendu à petit `n` |
| `lora` | adaptateurs de rang faible sur les blocs d'attention | régularisation par contrainte de capacité |
| `norm-only` | seuls les paramètres de normalisation | borne basse de capacité, très robuste à petit `n` |
| `scratch` | initialisation aléatoire, budget identique | contrôle négatif, donne le facteur d'économie |
| `full-ft + aug` | symétrie gauche/droite, bruit, élastique léger | levier le moins coûteux |

Les bras `lora` et `norm-only` répondent à l'intuition centrale du sujet : à
petit `n`, ce qui compte n'est pas la taille des données mais le **nombre de
degrés de liberté effectivement adaptés**. Je m'attends à ce que le classement
entre bras s'inverse le long de la courbe — `norm-only` meilleur à `n ≤ 16`,
`full-ft` meilleur à `n ≥ 128` — et le point de croisement est un résultat en
soi.

**Budget de calcul constant** : nombre de pas d'optimisation identique pour tous
les `n` et tous les bras. Sélection du point de contrôle par un critère qui
n'utilise pas le jeu de référence (perte de diffusion sur un jeu de validation
tiré du vivier), jamais par la métrique d'évaluation.

## 5. Évaluation — gelée pour tous les points de la courbe

| élément | valeur | pourquoi |
|---|---|---|
| `N_gen` | 512 volumes | identique partout, sinon le biais varie le long de la courbe |
| `N_ref` | tout `R` | idem |
| encodeur | encodeur BLAST-CT figé | sensibilité aux structures cliniquement pertinentes (cf. note 01 §2) |
| largeur de bande RBF | médiane calculée **une fois** sur `R` | sinon la métrique change d'échelle d'un point à l'autre |
| échantillonneur | DDIM, 50 pas, `η = 0`, graines fixées | reproductibilité |

**Métriques rapportées** (les quatre axes, jamais un seul chiffre) :

- adéquation : `MMD²` sans biais (principal), KID, FID et FID∞ (pour
  comparaison avec la littérature) ;
- fidélité : précision, densité ;
- diversité : rappel, couverture ;
- nouveauté : `AuthPct`, ratio de distance au plus proche voisin, taux de copie,
  **calibrés sur `H`** ;
- sémantique : `W₁` entre distributions de volume lésionnel, nombre de lésions,
  latéralité, type d'hémorragie — attributs estimés par BLAST-CT appliqué
  identiquement aux images réelles et générées.

**Lecture radiologique.** Test de Turing visuel sur 100 images (50 réelles,
50 générées, ordre aléatoire), 2 lecteurs indépendants, mesure de l'AUC de
détection et du κ inter-lecteur. C'est le critère de référence de la
littérature médicale, et le seul que le comité scientifique du CHU
considérera comme dirimant. À faire une seule fois, en fin de stage, sur les
2 ou 3 configurations retenues — c'est une ressource humaine rare.

## 6. Analyse statistique

1. Ajustement de `Q(n) = Q∞ + a·n^(−α)` par moindres carrés sur `log Q`
   (les erreurs sur une divergence sont multiplicatives) ;
2. bootstrap stratifié par `n`, 2 000 tirages, rééchantillonnant les
   répétitions → IC 95 % sur `α`, `Q∞`, `n*(τ)` ;
3. si une fraction non négligeable des tirages donne `n* = +∞`, elle est
   **rapportée comme telle** : c'est le résultat, pas une donnée manquante ;
4. facteur d'économie de données `n_scratch(q) / n_pretrained(q)` à plusieurs
   niveaux `q` — la quantité que le projet peut réellement utiliser ;
5. **analyse de sensibilité obligatoire** : refaire toute l'analyse avec
   (a) l'encodeur InceptionV3 à la place de BLAST-CT, (b) la FID à la place de
   la MMD². Si `n*` change d'un ordre de grandeur, c'est le résultat principal
   du stage — et il est publiable.

## 7. Ce qu'il faut pré-enregistrer avant le premier balayage

Découpages et graines · prétraitement · grille `n` et bras · `N_gen`, `N_ref`,
encodeur, largeur de bande · définition de `τ` · règle de sélection du point de
contrôle · plan d'analyse. Écrire ce fichier avant de lancer le premier
entraînement coûte une demi-journée et protège de la seule erreur vraiment
coûteuse de ce type d'étude : ajuster le protocole en regardant les résultats.

## 8. Ordonnancement et budget de calcul

| phase | contenu | calcul |
|---|---|---|
| A | prétraitement, découpages, encodeur, `τ` | CPU |
| B | zero-shot + reproduction de MAISI | ~2 j GPU |
| C | balayage `full-ft` et `scratch`, 8 `n` × 5 graines × 2 bras | ~80 exécutions |
| D | bras `lora`, `norm-only`, `+aug` | ~120 exécutions |
| E | sensibilité (encodeur, métrique), lecture radiologique | CPU + 2 lecteurs |
| F | réplication sur la cohorte GenAIPI-TBI | ~80 exécutions |

Une exécution = adaptation + échantillonnage de 512 volumes. C'est
l'échantillonnage qui domine en 3D : prévoir DDIM 50 pas et, si nécessaire, un
distillat de l'échantillonneur. Point de vigilance : la phase C seule
représente plusieurs semaines de GPU — le dimensionnement du calcul doit être
tranché en phase A, pas découvert en phase C.

## 9. Critères d'arrêt et résultats attendus

L'étude est concluante si elle produit, pour chaque bras :
`α` et son IC, `Q∞` et son IC, `n*(τ₉₅)` et son IC — **ou** la conclusion
argumentée que `τ` est sous le plancher, auquel cas la recommandation porte sur
la méthode d'adaptation et non sur la taille de cohorte.

Les deux issues sont des résultats. La seconde est même la plus utile au
projet : elle évite d'engager une collecte de données qui n'aurait pas résolu
le problème.
