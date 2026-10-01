# Détection d'endommagement par ondes guidées sous variation de température (Open Guided Waves, jeu n°2)

English version: [README.md](README.md)

Projet personnel : détecter un défaut sur une plaque composite instrumentée,
malgré des variations de température de 20 à 60 °C, avec trois approches
comparées sans fuite de données (une méthode classique de référence, un réseau
convolutif supervisé, une détection d'anomalie apprise sur des mesures saines
seules), puis compresser et exporter les modèles pour l'embarqué (ONNX,
quantification int8, génération de C++ avec Eclipse Aidge).

**Résultat principal.** Un réseau convolutif supervisé entraîné sur trois
positions de défaut **ne détecte pas** une quatrième position jamais vue
(AUC de 0,25 à 1,00 selon la position). Une analyse en composantes
principales à 5 composantes, apprise sur 79 mesures **saines** seulement,
détecte les défauts aux quatre positions (AUC 1,000), généralise à une plage
de température absente de l'apprentissage, et tient en 337 Ko avec des poids
int8, contre 34,85 Mo pour la bibliothèque de mesures de la méthode de référence.

## Données
Jeu n°2 de la plateforme Open Guided Waves : Moll J., Kexel C., Pötzsch S.,
Rennoch M., Herrmann A. S., *Temperature affected guided wave propagation in a
composite plate complementing the Open Guided Waves Platform*, Scientific Data
6, 191 (2019), <https://doi.org/10.1038/s41597-019-0208-1>. Données :
<https://doi.org/10.6084/m9.figshare.c.4488089.v1> (licence CC0, environ 160 Go,
non incluses dans ce dépôt).

- Plaque CFRP de 500 × 500 × 2 mm, 12 transducteurs piézoélectriques, soit
  **66 trajets** émetteur-récepteur (numérotation 0 à 11 du fichier HDF5).
- Défaut réversible : disque d'aluminium fixé en surface par un adhésif
  repositionnable (d'après la publication), à 4 positions (D04, D12, D16, D24).
- **966 mesures** : 322 saines (deux cycles de 161) et 161 par position de
  défaut. Chaque cycle : 20 → 60 → 20 °C par pas de 0,5 °C (consigne).
- 12 fréquences d'excitation (40 à 260 kHz), salves de 5 cycles ; signaux de 13 108
  points à 10 MHz (1,31 ms).
- Comme l'indique la publication, l'effet de la température sur le signal
  dépasse celui d'un défaut : c'est la difficulté centrale du problème.

## Données et prétraitement
1. **Centrage** de chaque signal (offset continu de 7 % du RMS à 40 kHz).
2. **Décimation** par 16 avec filtre anti-repliement : 625 kHz, **820 points**.
3. **Passe-bande** [f/2, 2f], soit [20, 80] kHz.
4. **Échelle par trajet**, estimée sur les mesures d'entraînement seules.
5. **Contrôle qualité** : 11 couples (mesure, trajet) sont quasi vides (tous
   sur le trajet 8-9) ; ils sont traités comme des pannes de capteur, pas
   comme des défauts.

**Température.** La consigne de l'enceinte n'est pas la température de la
plaque : l'écart moyen dépend de la série de mesures (+1,11 °C pour le premier cycle
sain, +0,55 °C pour le second, de +0,26 à +0,34 °C pour les séries
endommagées). Toutes les bandes de température portent sur la température
mesurée sur la plaque.

## Choix de la fréquence
Rapport entre le résidu dû au défaut et la variabilité entre deux cycles sains
(signaux centrés et filtrés, référence appariée sur la température mesurée ;
`tools/frequency_contrast.py`, tableau complet dans
`results/step01_exploration/frequency_contrast.md`) :

| kHz | 40 | 60 | 80 | 100 | 120 | 200 | 260 |
|---|---|---|---|---|---|---|---|
| médiane sur les trajets | **2,28** | 1,72 | 1,39 | 1,31 | 1,32 | 1,46 | 1,32 |
| maximum sur les trajets | **9,01** | 6,50 | 5,29 | 4,03 | 2,23 | 3,57 | 4,27 |

Fréquence retenue : **40 kHz**.

## Protocole : découpage sans fuite de données
L'unité est la **mesure** (ses 66 trajets vont toujours ensemble).
- **Principal, par position (4 plis)** : test = une position de défaut jamais
  vue + le second cycle sain (161 + 161) ; entraînement (580 mesures) et
  validation (64, bande 38 à 42 °C) sur les trois autres positions et le premier cycle sain.
- **Par température** : test = bande 48 à 52 °C, toutes classes.
- **Contrôle de dérive** : premier contre second cycle sain (toutes les
  mesures saines précèdent les mesures endommagées ; le modèle ne doit pas apprendre la date).
- Normalisation, références et seuils appris sur l'entraînement seul ;
  configuration figée sur la validation ; **test ouvert une seule fois par méthode**.
  Une exception : le contrôle de qualité d'acquisition de la référence
  (étape 2) a été ajouté après examen de deux fausses alarmes du test (cycle
  sain 2, capteur défaillant sur le trajet 8-9). Le même défaut existe dans les
  données d'entraînement : un contrôle mené sur l'entraînement seul l'aurait
  aussi révélé.
- **Validation imbriquée** pour le réseau supervisé (une position
  d'entraînement mise de côté, test externe écarté) : elle a prédit son échec avant l'ouverture du test.

## Résultats (AUC de test)
| Découpage | Référence (sans apprentissage) | CNN supervisé (5 graines) | ACP, k = 5 (sains seuls) |
|---|---|---|---|
| position D04 jamais vue | 1,000 | 1,000 ± 0,000 | 1,000 |
| position D12 jamais vue | 1,000 | 0,581 ± 0,113 | 1,000 |
| position D16 jamais vue | 1,000 | 0,247 ± 0,073 | 1,000 |
| position D24 jamais vue | 1,000 | 0,446 ± 0,122 | 1,000 |
| température jamais vue | 0,851 | 1,000 ± 0,000 | 1,000 |
| contrôle de dérive | | 0,555 ± 0,124 | |

- **Référence** : pour chaque mesure, la mesure saine de la bibliothèque la
  plus ressemblante (sélection de ligne de base optimale), résidu normalisé
  par trajet, maximum sur les trajets. Au seuil calibré sans le test : 100 %
  de détection, 1,2 % de fausses alarmes sur les positions, mais **64,7 %** sur
  la température jamais vue (aucune référence à ces températures).
- **CNN supervisé** (65 601 paramètres, 4 blocs Conv1d/BatchNorm/ReLU/MaxPool,
  maximum global) : il reconnaît les positions vues ; sur D16, il classe le
  défaut franchement sain (logit médian −9,5, contre +8,8 pour les positions
  vues). La régularisation ne change rien (validation imbriquée 0,647 contre 0,644).
- **ACP** : projection sur 5 composantes apprises sur 79 mesures saines (montée
  du premier cycle, moins 2 mesures à trajet vide) ;
  indicateur = résidu de projection, maximum sur les trajets. Au seuil
  calibré : 100 % de détection, 7,5 % de fausses alarmes (toutes entre 20,6
  et 24,0 °C, au bord froid du domaine appris), 0 % sur la température jamais vue.

## Localisation du défaut
Avec 4 positions de défaut alignées, un modèle supervisé ne peut pas
apprendre à localiser une position nouvelle : régression ridge ou perceptron
multicouche entraînés sur 3 positions, évalués sur la 4e, erreur médiane de
184 à 193 mm (sauf la position intermédiaire D16, interpolée à 24 mm). La
localisation vient donc de la géométrie (coordonnées des tableaux 1 et 2 de
la publication), à partir des résidus du modèle sain (ACP), sans aucune
position de défaut pour régler les paramètres :

| Erreur médiane (mm) | D04 | D12 | D16 | D24 | toutes |
|---|---|---|---|---|---|
| RAPID (amplitude par trajet) | 369 | 24 | 24 | 133 | 128 |
| Retard et somme (temps de vol, v = 1,284 mm/µs calibrée sur mesures saines) | 45 | 16 | 22 | 46 | 40 |

RAPID ne situe pas un défaut le long d'un trajet de bord (D04, D24) ; le
temps de vol le permet. Les estimations aux bords restent biaisées vers
l'extérieur (cause non établie ; la vitesse n'est pas en cause, ±5 % donne
38 à 55 mm).

## Autres fréquences
Même protocole aux 12 fréquences (40 à 260 kHz) : la chaîne ne fonctionne qu'à
basse fréquence. Détection par ACP : AUC 1,000 à 40 kHz, 0,996 à 1,000 à
60 kHz, proche du hasard au-delà de 100 kHz (plancher de résidu sain insensible
au nombre de composantes : limite de rapport signal sur bruit). Localisation :
40 mm à 40 kHz, 194 mm ou plus ailleurs (la calibration de vitesse par première
arrivée capte un autre mode au-delà de 40 kHz ; à 60 kHz l'ajustement échoue
franchement, avec une vitesse négative de -5,99 mm/µs, et à 260 kHz t0 est
négatif). La fusion des 12 fréquences dégrade la localisation (205 mm). Sur le
découpage par température, l'AUC de l'ACP est erratique au-delà de 60 kHz
(0,28 à 140 kHz, 0,99 à 200 kHz), sans tendance cohérente. Le choix de 40 kHz,
fait au départ avec les défauts des 4 positions, a été contrôlé par une
sélection sans fuite (validation de chaque pli seule) : 40 kHz a la meilleure
AUC de validation dans 3 plis ; dans le pli D16, 40 et 60 kHz sont à égalité
(AUC de validation 1,000). Détails : `results/step07_frequencies/`.

## Auto-encodeurs profonds
Même protocole que l'ACP (apprentissage sur les 79 mesures saines, latent
choisi sur la validation, 3 graines). Un auto-encodeur dense
(6 981 866 paramètres, 27,93 Mo) égale l'ACP sur les positions (AUC 1,000) ; un
auto-encodeur convolutif compact (85 900 paramètres) détecte moins bien (AUC
de 0,763 à 1,000 ; non convergé en 3 000 époques). Sur la température jamais
vue, leurs seuils se transposent mal (42 % et 100 % de fausses alarmes, contre
0 % pour l'ACP). L'ACP, auto-encodeur linéaire, reste le meilleur compromis.

## Embarqué : ONNX et quantification
| Modèle | Fichier ONNX | Latence (1 fil, lot de 1) | Détection |
|---|---|---|---|
| CNN float32 | 285 Ko | 0,24 ms | voir ci-dessus |
| CNN int8 statique (par canal) | 99 Ko | 0,07 ms | écart d'AUC ≤ 0,018 par modèle, ≤ 0,006 en moyenne sur 5 graines |
| ACP float32 | 1 285 Ko | 0,10 ms | AUC 1,000 |
| ACP, poids int8 | 337 Ko | 0,12 ms | AUC 1,000, 6,2 % de fausses alarmes |

La quantification int8 dynamique de l'ACP est écartée : fichier plus gros
(1 549 Ko, une copie float32 des composantes subsiste) et seuil non
transposable (94 % de fausses alarmes sur la température jamais vue).
Latences mesurées sur un processeur de station de travail : ordres de grandeur relatifs.
Non compté : le prétraitement (décimer depuis 10 MHz coûte plus que les
modèles ; en embarqué, on échantillonnerait directement plus lentement).

## Export C++ avec Aidge
Avec Eclipse Aidge 0.10.1, le C++ généré en float32 est fidèle : écart de
8e-6 sur le logit du CNN, écart relatif de 7e-4 sur l'indicateur ACP (64
mesures de test). Quatre réécritures exactes ont été nécessaires :
normalisation repliée dans la première convolution, format de données
explicite (le format automatique produit un programme qui plante), réécriture
2D du CNN (la quantification ne gère pas MaxPooling1D), ACP sans sortie booléenne.
**Int8 C++** : avec la configuration d'origine, le programme généré écrase la
sortie, alors que le graphe quantifié par Aidge est correct en Python
(corrélation 0,989 avec le float) et qu'ONNX Runtime quantifie le même graphe
sans perte (corrélation 0,995). Un cas minimal (une convolution + ReLU, 15 tailles) montre
qu'une **convolution int8 C++ n'est exacte que si ses nombres de canaux
d'entrée et de sortie sont multiples de 16**. Contournement exact : entrée
complétée de 66 à 80 canaux nuls, sortie de 1 à 16 ; le CNN int8 C++ devient
fidèle (corrélation 0,989, 68 Ko de paramètres). Détails : `results/step05_aidge_export/`,
`results/step09_aidge_int8_case/`, `step05_aidge_export/`,
`step09_aidge_int8_case/`.

## Sur microcontrôleur (Cortex-M7 émulé)
Compilation pour Cortex-M7 (arm-none-eabi-gcc 15.2.1, `-O3`) et exécution
sous QEMU (carte mps2-an500) ; **pas de carte réelle**. Tailles lues sur le
binaire ; instructions comptées avec `-icount` et calibrées sur une boucle
de longueur connue ; durées **estimées** à 480 MHz (une instruction par cycle).

| Détecteur | Flash | RAM statique + entrée | Instructions par mesure | ≈ ms à 480 MHz |
|---|---|---|---|---|
| ACP k = 5, poids int8 (C écrit à la main) | 353 Ko | 3 Ko + 211 Ko | 3,25 M | 6,8 |
| CNN int8 (Aidge, canaux complétés) | 104 Ko | 311 Ko + 64 Ko | 120,65 M | 251 |
| CNN float32 (Aidge) | 291 Ko | 317 Ko + 211 Ko | 107,89 M | 225 |

Sorties vérifiées : ACP à 1,9e-5 près de Python ; réseaux identiques au
programme hôte. Sans noyaux vectorisés (CMSIS-NN, non testés), l'int8 gagne
de la mémoire mais pas de temps.

## Limites
Une seule plaque, un défaut artificiel réversible, la température comme seule
nuisance, une seule campagne de mesure ; fausses alarmes au bord froid du
domaine appris ; microcontrôleur émulé seulement (pas de carte réelle). La sélection de ligne
de base optimale et la compensation de température sont des techniques
connues ; l'apport de ce projet est la comparaison contrôlée des trois
approches et le passage à l'embarqué.

## Reproduire
```
# torch==2.14.0+cpu provient de l'index CPU de PyTorch
python -m venv .venv && .venv/bin/pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e . --no-build-isolation
# données : télécharger les 5 fichiers zip de la collection figshare dans data/raw/
.venv/bin/python tools/build_index.py
.venv/bin/python tools/extract_frequency.py 40
# les 12 fréquences (environ 3,3 Go chacune), nécessaires à l'étape 7
for f in 60 80 100 120 140 160 180 200 220 240 260; do .venv/bin/python tools/extract_frequency.py $f; done
.venv/bin/python -m pytest -q
.venv/bin/python step01_exploration/exploration.py
.venv/bin/python tools/frequency_contrast.py > results/step01_exploration/frequency_contrast.md   # tableau du choix de fréquence
.venv/bin/python step02_baseline/evaluate_baseline.py
.venv/bin/python step03_cnn_vs_pca/train_cnn.py --phase validation
.venv/bin/python step03_cnn_vs_pca/train_cnn.py --phase validation --config regularized
.venv/bin/python step03_cnn_vs_pca/train_cnn.py --phase test
.venv/bin/python step03_cnn_vs_pca/evaluate_pca.py --phase validation
.venv/bin/python step03_cnn_vs_pca/evaluate_pca.py --phase test
.venv/bin/python step03_cnn_vs_pca/summary.py
.venv/bin/python step04_embedded_export/embedded_export.py
# Aidge : environnement séparé
python -m venv .venv-aidge && .venv-aidge/bin/pip install -r requirements-aidge.txt
.venv/bin/python step05_aidge_export/prepare_data.py
.venv-aidge/bin/python step05_aidge_export/export_aidge.py
.venv-aidge/bin/python step05_aidge_export/diagnostic_int8.py
.venv/bin/python step05_aidge_export/check_onnxruntime.py
.venv/bin/python step06_localization/localize.py
.venv/bin/python step07_frequencies/frequencies.py
.venv-aidge/bin/python step09_aidge_int8_case/minimal_case.py
.venv/bin/python step08_cortex_m7/build.py    # outils xPack arm-none-eabi-gcc et QEMU dans arm_tools/
.venv/bin/python step10_autoencoders/autoencoders.py
```

## Organisation
`src/ogw/` (`data` : données, `splits` : découpages, `baseline` : référence,
`models` : modèles, `training` : entraînement, `anomaly` : ACP, `embedded` :
embarqué, `localization` : localisation, `autoencoder` : auto-encodeurs) ;
`tests/` ; `step01_exploration/` à `step10_autoencoders/` (scripts des
étapes 1 à 10) ; `tools/` (extraction, diagnostics) ; `results/` (figures,
tableaux, métriques).

## Note
Le code a été écrit avec l'assistance d'un outil d'IA (Claude, Anthropic).

## Licence
Code sous licence MIT (voir LICENSE). Les données Open Guided Waves sont
distribuées sous licence CC0 par leurs auteurs et ne sont pas incluses.
