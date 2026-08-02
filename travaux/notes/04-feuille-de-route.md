# Feuille de route indicative — 6 mois (avril → septembre 2026)

Découpage en jalons vérifiables, avec pour chacun un livrable qui existe ou
n'existe pas — pas « avancer sur X ». Les durées supposent un stagiaire M2 en
mathématiques appliquées qui connaît PyTorch mais pas l'imagerie médicale.

---

### M1 — Avril : instrumentation avant modélisation

| semaine | livrable |
|---|---|
| 1 | environnement, accès CQ-500, lecture du sujet et des 4 références clés |
| 2 | chaîne de prétraitement figée et versionnée ; découpages `R`/`H`/`P` gelés |
| 3 | batterie de métriques **avec ses tests de validation** (cf. `tests/`) |
| 4 | seuil `τ₉₅` calibré ; protocole pré-enregistré (`02-protocole.md` instancié) |

> Le piège de ce mois est de vouloir « lancer un premier entraînement pour
> voir ». Une métrique non validée produit des courbes qui semblent
> interprétables et ne le sont pas ; le coût de la découvrir en juillet est un
> mois de balayage à refaire. La batterie de tests de la semaine 3 est
> l'investissement le plus rentable du stage.

**Point de contrôle fin M1** : le générateur-copieur (celui qui rééchantillonne
son jeu d'entraînement) doit obtenir un excellent score d'adéquation *et* être
démasqué par l'audit de nouveauté. Si ce n'est pas le cas, la batterie n'est pas
prête et le balayage attend.

### M2 — Mai : reproduction et contrôles

| semaine | livrable |
|---|---|
| 5 | MAISI installé, inférence reproduite, latents calculés sur `R` |
| 6 | **`n = 0`** : évaluation zero-shot complète — combien de la cible est déjà couverte ? |
| 7 | boucle d'adaptation `full-ft`, une exécution complète bout en bout |
| 8 | dimensionnement du calcul de la phase C ; décision sur la grille finale |

**Point de contrôle fin M2** : `Q(0)` est mesuré et l'étude est dimensionnée.
Si `Q(0)` est déjà proche de `τ`, le sujet bascule vers la question
conditionnelle plus tôt que prévu — c'est une bonne nouvelle, pas un échec, mais
elle doit être prise en mai et pas en août.

### M3 — Juin : le balayage principal

| semaine | livrable |
|---|---|
| 9-10 | balayage `full-ft` × `scratch`, 8 valeurs de `n`, 5 graines |
| 11 | ajustement de la loi d'échelle, bootstrap, premier `n*(τ)` avec IC |
| 12 | audit de mémorisation sur toute la grille ; premières planches d'échantillons |

**Point de contrôle fin M3** : une courbe `Q(n)` avec IC, un `α`, un `Q∞`, un
`n*`. C'est le cœur du mémoire ; s'il n'existe pas fin juin, la suite est
compromise et il faut réduire le périmètre (moins de bras, pas de réplication).

### M4 — Juillet : d'où vient l'efficacité

| semaine | livrable |
|---|---|
| 13-14 | bras `lora`, `norm-only`, `+aug` ; courbes comparées |
| 15 | point de croisement entre bras ; recommandation de méthode par régime de `n` |
| 16 | analyse de sensibilité : encodeur BLAST-CT vs InceptionV3, MMD² vs FID |

**Point de contrôle fin M4** : on sait dire « pour `n < N₀`, adapter seulement
les normalisations ; au-delà, adapter tout », avec `N₀` chiffré. C'est la
recommandation opérationnelle que le projet attend.

### M5 — Août : théorie et réplication

| semaine | livrable |
|---|---|
| 17 | dimension intrinsèque des latents (Levina–Bickel, TwoNN) sur CQ-500 |
| 18 | confrontation `α` observé / prédiction `2s/(2s+d_int)` (cf. note 01 §7) |
| 19-20 | réplication sur la cohorte GenAIPI-TBI ; comparaison des deux `n*` |

C'est le mois qui décide si le mémoire est un rapport de stage ou un article.
Le contraste entre `n*` sur données publiques et `n*` sur données non publiées
est, en soi, un résultat sur la contamination des jeux d'évaluation. À aborder
seulement si les jalons M3 et M4 sont tenus.

### M6 — Septembre : conditionnel et rédaction

| semaine | livrable |
|---|---|
| 21 | proxy conditionnel : génération conditionnée par un masque lésionnel |
| 22 | même méthodologie `n*` appliquée au conditionnel — premier chiffre |
| 23-24 | mémoire, soutenance, dépôt du code et des protocoles |

Si le temps manque, c'est la semaine 21-22 qui saute : elle ouvre la suite du
projet mais ne conditionne pas la validité de ce qui précède.

---

## Risques et parades

| risque | probabilité | parade |
|---|---|---|
| calcul GPU insuffisant pour la phase C | élevée | dimensionner en M2 ; réduire `N_gen`, la grille ou passer en 2,5D |
| CQ-500 déjà dans le corpus de MAISI | moyenne | réplication sur GenAIPI-TBI (M5) ; à documenter dans tous les cas |
| `τ` sous le plancher `Q∞` pour tous les bras | moyenne | c'est un résultat ; bascule vers la comparaison de méthodes |
| accès tardif aux données grenobloises | moyenne | M5 semaine 19-20 est le seul jalon concerné, déplaçable en M6 |
| stagiaire bloqué sur MAISI (dépendances MONAI, 3D) | élevée | prévoir une réplique 2D « jouet » dès M1 pour dérisquer la chaîne complète — c'est exactement le rôle de `src/fewshotgen/` dans ce dépôt |

## Ce que le stage produit dans tous les cas

Même dans le pire scénario (MAISI inutilisable, calcul insuffisant), le stage
livre : une chaîne d'évaluation validée et réutilisable par le projet, un seuil
`τ` calibré sur les données du CHU, un audit de mémorisation opérationnel — qui
sera de toute façon exigé par le comité d'éthique dès qu'il sera question de
partager des images synthétiques — et un protocole pré-enregistré. Ce sont des
livrables d'infrastructure dont GenAIPI-TBI aura besoin quelle que soit la
suite.
