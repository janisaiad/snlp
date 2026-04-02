# Script oral — ML-SUPERB et statut du volet JEPA

Narration complète pour le diaporama Beamer **report_beamer.pdf** (piste Janis : conduite des entraînements, résultats, analyse des échecs). Tu peux lire section par section pendant la présentation. Les titres de slides correspondent au PDF.

---

## Slide 1 — Page de titre

Bonjour à toutes et à tous. Je suis Janis Aiad, avec l’équipe, de l’ENS-PSL.

Cette présentation fait le **point d’avancement** sur notre travail autour de **ML-SUPERB** et de l’apprentissage de **représentations façon JEPA** pour la parole. Le sous-titre résume ce que je vais vraiment traiter : comment nous avons **piloté les entraînements**, quels **résultats** nous avons obtenus, et comment nous avons **analysé les échecs** — y compris lorsque les métriques se comportent de manière **trompeuse**.

Je ne présuppose pas que tout le monde connaisse chaque détail de MLSUPERB : voyez cela comme un **benchmark ASR très basse ressource** où l’on **gèle** un **frontend** auto-supervisé et où l’on n’entraîne qu’une petite **tête CTC** sur une **quantité minuscule** de parole étiquetée.

**Remarque pratique :** les **titres des slides sont en anglais** (terminologie MLSUPERB, ESPnet, cours cités en VO) ; tu **commentes en français** — c’est volontaire, pas une erreur.

---

## Slide 2 — Objectif de recherche (Janis)

**Protocole.** Notre référence est le protocole d’évaluation **ML-SUPERB**. Nous **gelons** le frontend en apprentissage auto-supervisé — dans les expériences principales, nous ne fine-tunons pas le gros transformateur sur les étiquettes ASR. Au-dessus de cet encodeur figé, nous entraînons une **tête CTC légère**. Les budgets étiquetés sont volontairement **dérisoires** : **10 minutes** de parole et **une heure** de parole. C’est tout l’enjeu du benchmark : **tester les représentations** sous **ressource extrêmement limitée**.

**Ce que nous comparons.** Nous gardons la **recette en aval aussi identique que possible** d’un modèle à l’autre, et nous changeons la **famille de représentations** :

1. **HuBERT**, chargé via **S3PRL**, gelé — notre **baseline forte et reproductible**.
2. Un montage **JEPA minimal** — encodeur de style JEPA **allégé**, sans la même **échelle de pré-entraînement** que WavJEPA.
3. **WavJEPA** sous deux formes : poids issus de la version **publique Hugging Face**, versus un **checkpoint que nous avons pré-entraîné localement** avec le code officiel WavJEPA et des données de type AudioSet.

La question scientifique n’est donc pas « quel outil gagne » mais bien : **à recette CTC et budgets identiques, quelles représentations figées portent la structure phonétique et lexicale la plus utile ?**

**Réalité ingénierie.** Avant de pouvoir faire confiance à une grille d’expériences, il a fallu **monter tout le pipeline** : préparation des données, configurations ESPnet, décodage, scoring, sauvegarde des checkpoints, lancements reproductibles. **Stabiliser tout ça de bout en bout** a pris environ **une journée de travail pleine**. Je le signale parce qu’une bonne partie de nos « échecs » dans les journaux relève de **l’infrastructure et de l’hygiène des reprises**, pas d’un effondrement du modèle.

**Calcul.** Nous avons utilisé un **contexte matériel mixte** :

- Les **jobs lourds** — en particulier le **pré-entraînement WavJEPA sur AudioSet** et les **longues reprises de checkpoint** — ont tourné sur un **NVIDIA H100** avec **80 Go** de mémoire GPU. La machine hôte disposait d’une **grande RAM système**, de l’ordre d’**environ 80 Go**, ce qui compte pour le **chargement des données**, le tamponnage et les **E/S de checkpoints** lorsqu’on n’est pas limité par un petit nœud.
- La **majorité du fine-tuning ML-SUPERB** et les **chaînes de balayage orchestrées** ont tourné sur **une seule GPU NVIDIA L4**. Même recette ESPnet, mais **empreinte VRAM plus faible** par job ; nos synthèses de temps sont **dominées** par ces expériences sur L4.

**Extensions.** Au-delà de l’anglais monolingue, nous avons étendu le même schéma à l’**ASR multilingue agrégé**, à l’**ASR conjoint avec identification de langue (ASR+LID)** et au **fine-tuning efficace en paramètres** avec **LoRA** — toujours dans le cadre **basse ressource** à la MLSUPERB.

---

## Slide 3 — Question directrice : ABX vs ASR, JEPA vs SSL masqué

Je précise tout de suite la **ligne Poli / Dupoux** pour que l’auditeur sache **ce que la grille ASR teste vraiment**.

**Première puce du PDF :** enseignement du cours : les frontends auto-supervisés peuvent bien encoder une **structure phonétique fine** — ce que l’**ABX** met en évidence — sans que le **classement** soit **identique** sur la **transcription lexicale** mesurée en **CER/WER**, avec **même** recette CTC et **même** budget étiqueté. C’est ça, la **tension** centrale.

**Deuxième puce :** nous mettons en face **JEPA / WavJEPA** — cibles latentes **continues** et prédiction temporelle — et **HuBERT** — prédiction sur **tokens masqués** ou **clusters discrets**. La grille MLSUPERB demande si **la forme du signal d’apprentissage en amont** se **propage pareil** jusqu’au CTC **mot**.

**Troisième puce :** dans **nos** chiffres eng1 10 min, **HuBERT** peut être **meilleur en ABX** que **WavJEPA HF** tout en restant **à égalité en CER/WER** → une **seule** métrique ne suffit pas. Ça **ancre** la question d’ouverture **sur un cas concret** du projet, pas seulement sur la théorie du cours.

---

## Slide 4 — Brief complet de recherche vs portée Janis (limites et suite)

Cette slide est pour un **jury qui attend un « papier »** : elle **nomme** ce qui est **fait**, ce qui est **hors scope**, et ce qui serait le **prochain bloc publishable**.

**Livrés sur la piste Janis :** trois **familles** de frontends (HuBERT, JEPA minimal, WavJEPA), comparaison **HF vs checkpoint local** pour WavJEPA, branches **multilingue ASR / ASR+LID / LoRA**, **ABX** sur la **tranche dev 311 énoncés**, analyses ciblées (**export speech\_encoder**, **géométrie des poids CTC**).

**Lacune nette pour une matrice « article » très serrée :** une **deuxième** grande famille SSL **mature** sur la **même** grille complète — typiquement **HuBERT + XLSR-53** ou classe **wav2vec2** — n’est **pas bouclée** ici. Si le jury veut **deux** lignes SSL comparables **partout**, il faut le dire comme **travail suivant**, pas comme oubli implicite.

**Troisième axe du brief projet (SpidR) :** **désentrelacement / SpidR-Adapt** est **volontairement** hors piste Janis → on le cite comme **extension** pour ne pas donner l’impression d’une **preuve manquante** dans ce deck.

**Histoire par couche :** ABX et ASR **couche par couche** — le complément naturel aux scores **agrégés** — n’est **pas** le cœur de ce rapport ; on a seulement des **indices qualitatifs** côté **SER** en annexe Bruny. **Next** explicite si on pousse vers un **article** : cartographier **phonétique vs lexique** en profondeur.

---

## Slide 5 — Entraînements exécutés (inventaire)

Cette slide est un **catalogue** de ce que nous avons réellement exécuté ; elle complète la **ligne du temps** plus courte de la slide suivante.

**CTC monolingue, frontend gelé.** Nous avons lancé l’**anglais eng1** à **10 minutes** et **1 heure** avec **HuBERT**, **JEPA minimal**, **WavJEPA Hugging Face** et **WavJEPA checkpoint local**. Nous avons lancé le **français fra1** et l’**allemand deu1** à **10 minutes** et **1 heure** avec **HuBERT** pour compléter la **matrice par langue**. Sur **eng1**, nous avons aussi une **grille étendue** avec des expériences de comparaison sur **30 époques** pour **certains frontends** — c’est un régime de fine-tuning **plus long** et **plus stable** que les réglages MLSUPERB les plus courts.

**Multilingue agrégé.** Nous avons exécuté l’entraînement **ASR seul** à **10 minutes** et **1 heure**. Nous avons exécuté **LID seul** — identification de langue comme objectif séparé — à **10 minutes** et **1 heure**. **ASR+LID** à **10 minutes** est **terminé**. **ASR+LID** à **1 heure** a été **lancé** puis **arrêté** — je relierai cela plus loin aux problèmes de file et de reprise. **LoRA**, **ASR seul**, **10 minutes** est **terminé**. **LoRA**, **ASR seul**, **1 heure** a été **lancé**.

**Pré-entraînement WavJEPA.** Nous avons entraîné avec l’**objectif JEPA AudioSet** ; pour le long fil narratif, ce n’est pas « tout depuis l’init aléatoire en une seule traite » : nous avons **repris** depuis une **identité de checkpoint antérieure** et avons progressé jusqu’à **l’ordre de 174 mille pas globaux** dans les exécutions Lightning journalisées, en configuration **mono-GPU** dans ces journaux. Nous avons aussi fait tourner des **tests fumée** et des **briques d’automatisation longue** pour que le pipeline puisse tourner sans surveillance constante.

**Orchestration et analyse.** Nous avons enchaîné des **chaînes de benchmark** : phases **de mise au point** rapides d’abord, puis phases **nuit** ou **longues**. Nous avons **agrégé les métriques ASR** entre expériences. Nous avons exécuté l’**ABX** de discriminabilité sur une **partition de développement de 311 énoncés** ; cela a exigé de **corriger des outils tiers** — en l’occurrence l’interaction **DTW / torchdtw** avec l’**ABI PyTorch** actuelle. Après des **incidents de vocabulaire et de reprise**, nous avons fait des **passes de reproduction ciblées** pour séparer « expérience cassée » et « mauvais modèle ».

---

## Slide 6 — Ligne du temps (vue d’ensemble)

En un coup d’œil, chronologiquement :

Nous avons **terminé** la **matrice de référence monolingue eng1, 10 minutes** : HuBERT, JEPA minimal, WavJEPA Hugging Face, WavJEPA checkpoint local.

Nous avons exécuté un **arbre multilingue partiel** sur les langues disponibles — **anglais, allemand, français** — pour **ASR seul** à **10 minutes** et **1 heure**.

**ASR+LID** à **10 minutes** est **achevé**. La variante **1 heure** a **démarré** mais a été **interrompue**.

**LoRA** multilingue à **10 minutes** est **achevé** ; **1 heure** a **démarré** puis été **interrompue**.

Le **pré-entraînement WavJEPA AudioSet** a **repris** depuis un checkpoint et a atteint environ **174,6 k pas** avant interruption.

Nous avons **intégré et débogué l’ABX**, avec un **contournement** pour **fastabx** et **torchdtw**.

Donc le bilan : **couverture large**, mais **toutes les cases** d’une grille complète hypothétique ne sont pas vertes — certaines sont **arrêtées par des problèmes d’exploitation**, pas par une condition d’arrêt scientifique.

---

## Slide 7 — Monolingue eng1, 10 min (valeurs de référence)

Ce tableau est la comparaison **la plus propre** sur l’**anglais**, **10 minutes** d’étiquettes, **frontend gelé**, **CTC**.

Lecture des lignes :

- **HuBERT gelé :** **CER 33,33 %**, **WER 24,14 %**.
- **JEPA minimal :** **CER 62,22 %**, **WER 44,83 %** — nettement **plus mauvais**.
- **WavJEPA Hugging Face :** **CER 33,33 %**, **WER 24,14 %** — **identique** à HuBERT sur cette partition dans ce run.
- **WavJEPA checkpoint local, meilleur validé :** encore **CER 33,33 %**, **WER 24,14 %**.

**Synthèse en une phrase :** **HuBERT** et **WavJEPA** peuvent **s’aligner** sur ce réglage anglais basse ressource **lorsque WavJEPA est suffisamment pré-entraîné** — que ce soit avec les poids publics ou avec notre pré-entraînement local, on tombe dans un bon bassin. L’encodeur **JEPA minimal** n’est **pas** dans le même régime ; il **sous-performe fortement**, ce que nous attribuons à un **pré-entraînement insuffisant** pour cette recette en aval, et non à une comparaison équitable « JEPA vs HuBERT » à grande échelle.

---

## Slide 8 — Multilingue et adaptations (arbre MLSUPERB partiel)

Ici, pour ces chiffres multilingues agrégés, le **frontend** reste dans la lignée **HuBERT / S3PRL** — le tableau résume les réglages **multilingues** et d’**adaptation**, pas la grille WavJEPA complète sur chaque ligne.

**ASR seul, 10 minutes :** **CER 24,96 %**, **WER 23,48 %**.

**ASR seul, 1 heure :** **CER 20,76 %**, **WER 18,30 %** — **gain net** grâce à plus de données étiquetées.

**ASR+LID, 10 minutes :** **CER 26,33 %**, **WER 25,49 %** — **moins bon** que l’ASR seul au **même budget**. Ajouter l’**identification de langue** comme objectif conjoint **dégrade** un peu la transcription pure dans ce régime **basse ressource** ; c’est un **compromis**, pas forcément un bug.

**LoRA, ASR seul, 10 minutes :** **CER 24,95 %**, **WER 23,66 %**. Le **CER** est **à peu près identique** à l’ASR seul figé ; le **WER** est **légèrement plus mauvais** que la baseline figée à 23,48 %.

**Interprétation.** D’abord, **agréger les langues** et passer à **1 heure** aide beaucoup — on le voit sur la paire de lignes ASR seul. Ensuite, le **multitâche ASR+LID** a un **coût** sur l’ASR ici. Troisièmement, **LoRA** à 10 minutes est **compétitif en CER** mais ce n’est pas un **gain gratuit** en **WER** sur cet instantané.

---

## Slide 9 — Autres langues et statut de la file de jobs

Nous **avons bien** exécuté des expériences **HuBERT monolingues** pour l’**allemand deu1** et le **français fra1** à **10 minutes** et **1 heure** — ces chiffres alimentent la matrice élargie et les gammes de durées monolingues plus loin.

Nous avons aussi lancé une **longue file d’orchestration** de jobs. Elle s’est **arrêtée** à cause d’un **décalage de reprise**, pas parce que nous avions décidé que le modèle avait convergé.

Concrètement : la **dimension de sortie CTC** ne correspondait plus lorsque nous tentions de **reprendre** depuis un **ancien checkpoint** après **changement de liste de tokens** ou de **vocabulaire**. Un motif d’erreur typique : **45 classes CTC** dans le checkpoint obsolète contre **57** dans la configuration courante.

Je veux être clair : c’est un problème d’**hygiène de pipeline** et de **tenue de cahier d’expériences**. Ce n’est **pas** une preuve qu’une représentation est mauvaise. Dans l’exposé, nous **séparons** « la file a cassé » et « le modèle sous-performe ».

---

## Slide 10 — Résultats ABX (protocole actuel)

**ABX** est une sonde de **discriminabilité** : étant donnés trois segments A, B, X, la représentation place-t-elle X avec le bon « même classe » ? Pour un protocole fixé, une **erreur plus faible** est meilleure.

---

### FastABX : est-ce qu’on l’a utilisé ? Pourquoi ?

**Pourquoi c’est plus rapide (une ligne) :** fastabx remplace les boucles naïves triplet par triplet par des **distances vectorisées / par batch** et un **DTW efficace**, ce qui rend le scoring sur tout un dev faisable en temps raisonnable.

**Oui, nous nous appuyons sur fastabx** (bibliothèque associée au travail sur l’ABX efficace, Poli et co.) comme **moteur de calcul** pour estimer l’erreur ABX après extraction des représentations. **Pourquoi fastabx plutôt qu’un script maison ?** D’abord **cohérence** avec l’écosystème du laboratoire et la littérature récente sur l’ABX ; ensuite **implémentations visant l’efficacité** sur le protocole ABX (dont des parties qui passent par du DTW), ce qui évite de réimplémenter à la main un empilement fragile et lent. En résumé : **même définition de métrique** que dans les articles de référence, **moins de risque d’erreur** sur le scoring, et **temps de calcul** raisonnable sur des centaines d’énoncés.

**Pourquoi cela a « cassé » un moment ?** La chaîne fastabx s’appuie sur **torchdtw** pour certaines opérations ; avec **notre** version de **PyTorch**, une **incompatibilité binaire (ABI)** est apparue — ce n’était **pas** un rejet scientifique de l’ABX ni un problème HuBERT/WavJEPA, mais un **conflit de stack**. Nous avons mis en place un **contournement local** (correctif d’environnement ou de dépendance) ; **d’où la slide** : « FastABX required a local workaround ». **État actuel annoncé dans le deck** : l’**extraction + scoring** tourne **de bout en bout** sur la partition **dev** des **311 énoncés** ; les **chiffres** HuBERT vs WavJEPA ci-dessous proviennent de ce protocole **une fois** la chaîne réparée. (Sur l’historique du projet, une **phase automatisée** de benchmark a pu **échouer** sur ce point avant le correctif — à distinguer des **résultats finaux** affichés.)

---

Nous faisons donc **extraction et scoring de bout en bout** sur la partition **dev** — les mêmes **311 énoncés** que pour d’autres analyses.

**Chiffres sous le protocole actuel :**

- **Erreur ABX HuBERT :** **0,5216**
- **Erreur ABX WavJEPA Hugging Face :** **0,5895**
- **Delta** WavJEPA moins HuBERT : **+0,0679** — WavJEPA est **moins bon** sur cette métrique.

**Interprétation.** Dans **cette** configuration ABX, **HuBERT** est **plus discriminatif** au sens mesuré par l’ABX. Cela **ne contredit pas** la **parité CER/WER** observée plus tôt sur eng1 10 min : **ASR** et **ABX** sollicitent une **géométrie différente**. On peut avoir une **erreur de mots comparable** tout en **organisant autrement** le voisinage phonétique dans l’**espace des embeddings**. C’est exactement pourquoi nous suivons **à la fois** les métriques de **tâche** et les métriques de **représentation**.

---

## Slide 11 — Volet pré-entraînement WavJEPA

Le **gros** run WavJEPA n’était **pas** « entraînement from scratch dans un job ininterrompu depuis le pas zéro ». Il a **repris** depuis une **identité de checkpoint antérieure** — même lignée d’entraînement, optimisation poursuivie.

Nous sommes montés à l’ordre de **174,6 milliers de pas** avant **interruption** — là encore, exploitation ou planning, pas forcément le nombre de pas cible atteint.

**Volet de recherche en parallèle :** nous avons ajouté une ligne **from scratch** avec un profil **petit modèle** : taille **small**, **six** couches transformeur, **d\_model 384**. De **nouveaux scripts** raccordent le **pré-entraînement from scratch** aux **benchmarks ASR locaux**, et un **script de pipeline** peut tourner en mode **fumée** ou **long** pour l’automatisation.

En résumé : **poursuivre le gros pré-train repris**, tout en **montant** un régime **from scratch plus petit et contrôlé** pour des résultats scientifiques plus propres ensuite.

---

## Slide 12 — Pourquoi la loss baisse alors que le CER/WER reste mauvais

**Titre :** l’**objectif de pré-entraînement** et l’**objectif ASR en aval** ne sont **pas en correspondance directe** (on ne peut pas les aligner terme à terme).

Développons chaque puce :

1. Le pré-entraînement **façon JEPA** minimise l’**erreur de prédiction latente** entre vues contexte et cible. Il **ne** minimise **pas** le **taux d’erreur caractères** ni le **taux d’erreur mots**. Donc la **loss** peut **baisser** pendant que la **transcription** reste mauvaise.

2. Une **loss SSL plus basse** peut signifier que le modèle a appris de **bonnes invariances** pour son jeu **auto-supervisé** — masquage, prédiction, etc. — tout en gardant une **séparabilité lexicale** **faible** pour la **CTC** avec des **étiquettes minuscules**.

3. Avec **très peu** de supervision, la **tête CTC** peut **sous-apprendre** — pas assez de pas ou de paramètres — ou **sur-apprendre** — mémoriser des artefacts — ou **échouer à exploiter** les features figées si leur géométrie est mal adaptée à des décisions « presque linéaires » de type CTC.

4. Les expériences **de mise au point** avec **très peu d’époques ou d’itérations** peuvent donner des **CER/WER bruités ou non significatifs** même quand la **courbe de loss** est lisse. Ne sur-interprétez pas un **unique décodage exploratoire**.

5. **Décalage de distributions :** le pré-entraînement **AudioSet** vise de l’**audio général**. Les tranches **MLSUPERB** sont de la **parole lue** dans des langues précises. La loss SSL peut **s’améliorer** sur la distrib de prétrain **sans** gains **proportionnels** en **ASR**.

6. **Pollution des reprises** — **checkpoints périmés**, **listes de tokens modifiées** — peut faire **mentir** les courbes d’évolution. Vérifiez toujours l’**identité d’expérience** avant de comparer le pas *k* au pas *k+1*.

---

## Slide 13 — Liste de diagnostic pratique

Quand quelque chose a l’air faux, nous utilisons cette liste :

- **Isoler les expériences :** une **config**, une **langue**, répertoire d’expérience **propre** ou **étiquettes uniques** — pas de reprise silencieuse dans le mauvais dossier.
- Comparer les checkpoints avec le **protocole en aval complet**, pas un **décodage raccourci** depuis un cahier exploratoire.
- Suivre **à la fois** l’**ABX** (représentation) **et** **CER/WER** (tâche) pour détecter un **désalignement d’objectifs**.
- Pour les **tests A/B**, figer **graines**, **splits** et **réglages de décodage**.
- Dans les **rapports**, étiqueter séparément **échecs de pipeline** et **échecs de modèle** pour que personne ne confonde un **bug de dimension CTC** avec « WavJEPA ne marche pas ».

---

## Slide 14 — speech\_encoder vs ESPnet MLSUPERB (périmètre)

**Contexte :** le dépôt [**speech\_encoder**](https://github.com/iliasslasri/speech_encoder) (fork Textless) offre surtout une API **HuBERT + K-means** pour des **unités discrètes**, pas le flux **features continues $\rightarrow$ CTC** de la recette ESPnet MLSUPERB standard.

**Goulot sur nos runs MLSUPERB :** le temps est surtout mangé par l’**entraînement ASR** (CTC, décodage, E/S), pas par un **seul forward** HuBERT isolé. **speech\_encoder** ne remplace donc pas ce pipeline ; le README promet surtout **ergonomie** et usage **ciblé** (extraction d’unités, batch GPU avec longueurs), pas un **speedup massif** sur le fine-tuning CTC **tel quel**.

**Où un vrai gain de temps aurait pu apparaître :** **pré-calculer** une fois unités (ou embeddings) sur un corpus puis **réutiliser** pour plein de petites expériences — ou une **petite tête** sur IDs précomputés sans refaire l’encodeur à chaque run — mais cela **change** le protocole si MLSUPERB impose des **features continues** S3PRL côté ESPnet, ou exige un **apprentissage en aval** compatible.

**Ce qu’on n’a pas raté pour l’objectif principal :** WavJEPA, comparaisons HF vs local, LoRA, multilingue vivent dans **ESPnet + S3PRL / Hugging Face** ; speech\_encoder ne couvre ni préparation des données, ni décodage, ni scoring, ni hygiène du vocabulaire CTC.

Le **tableau** de la slide résume : pour **MLSUPERB CTC continu**, notre stack est le **chemin naturel** ; speech\_encoder est **très adapté** aux **unités Textless** ; pour la **vitesse à recette identique**, l’écart vs S3PRL est **≈ nul** — les bénéfices sont surtout **pédagogiques** et **méthodologiques** (alignement cours, expériences parallèles sur unités discrètes), **pas** un facteur deux sur le CTC actuel sans **refonte** de la tâche.

---

## Slide 15 — Section technique : données et protocole

**Recette :** recette **ASR** MLSUPERB sous **ESPnet** — **frontend SSL gelé**, tête **CTC**, plumbing de données MLSUPERB standard.

**Partitions :**

- **Monolingue :** **eng1**, **fra1**, **deu1** avec budgets **10 min** et **1 h**.
- **Arbre multilingue partiel :** ressources **anglais, allemand, français** agrégées quand disponibles — modes **ASR seul**, **LID seul**, **ASR+LID**, **LoRA**.
- **Analyse des représentations :** partition de **développement**, **311 énoncés**.

**Comparaison WavJEPA :**

- Checkpoint public **WavJEPA-base** sur **Hugging Face**.
- Checkpoint **pré-entraîné localement** sous **Lightning** — même **famille d’architecture**, **trajectoire de pré-entraînement** et gestion des données différentes.

**Runtime (rappel) :** nœud **classe H100** pour les **longs pré-trains** ; **L4** mono-GPU pour la **majorité du fine-tuning en aval** qui alimente nos **agrégats de temps**.

---

## Slide 16 — Section technique : temps d’entraînement mesuré

Nous lisons le temps **d’entraînement seul** dans les synthèses ESPnet — les lignes de **temps écoulé** en fin de jobs réussis, pas tout le pipeline.

**Ordres de grandeur côté L4 pour l’aval (heures) :**

- **ASR multilingue :** **~5,93 h** pour budget **10 min**, **~11,22 h** pour **1 h**.
- **LID seul :** **~4,93 h** en **10 min**, **~11,17 h** en **1 h**.
- **Matrice monolingue fra1/deu1 :** de l’ordre de **5,2 h à 14,7 h** selon le réglage.
- **WavJEPA en aval** sur **eng1 10 min :** **Hugging Face ~1,99 h**, **checkpoint local ~1,62 h**.

**Somme :** sur les exécutions en aval **réussies** journalisées dans cette phase, nous cumulons environ **78,3 heures-GPU** côté **L4**. Le pré-train sur **H100** ajoute du **temps mural** **en plus** et **n’est pas** inclus dans ce **78,3**.

Le **temps d’expérience total** est plus grand : **préparation des données**, **statistiques**, **décodage**, **scoring** ajoutent de la marge.

---

## Slide 17 — Section technique : estimation des FLOPs totaux

Nous n’avons pas de traces de **profiler** au niveau noyau pour **chaque** job, donc nous donnons une **estimation bornée** à partir des **heures-GPU** multipliées par un **débit effectif** en TFLOP/s.

Formule sur la slide : **FLOPs ≈ heures-GPU × 3600 × débit effectif (TFLOP/s)**.

Avec **78,3 heures-GPU** et un **pic tensor FP16 L4** comme **référence de borne supérieure théorique**, la slide indique de l’ordre de **~3,4 × 10^19 FLOPs** en **majoration** — c’est « si le GPU atteignait le pic **chaque** seconde », ce qui n’arrive pas.

Avec une **utilisation réaliste**, souvent **20 % à 40 %** du pic sur ces charges, la slide propose une **fourchette pratique** autour de **(0,7 à 1,4) × 10^19 FLOPs**.

**Message :** même avec un **frontend gelé**, l’ASR **basse ressource** à cette **ampleur de balayage** n’a **rien de gratuit** en calcul — la **largeur de la grille** se cumule.

---

## Slide 18 — Section technique : dérive checkpoint HF vs local

Nous avons comparé **WavJEPA Hugging Face** et **WavJEPA pré-entraîné localement** sur la partition **dev** de **311 énoncés** ; un **rapport écrit séparé** documente la méthode complète.

**Embeddings par énoncé :**

- **Similarité cosinus moyenne** entre moyennes HF et local : **−0,0082**, écart-type **0,0317** — ce n’est **pas** identique ; il y a de la **dispersion** selon les énoncés.
- **Rapport des normes** local sur HF, moyenne **1,5940** — les représentations locales sont **systématiquement plus grandes** en norme L2 en moyenne.

**Proxy au niveau mot :** les **centroïdes** par groupes de mots (premier jeton) montrent le **plus fort décalage** sur les **mots-outils fréquents** — là précisément où le regroupement peut **brouiller** la géométrie.

**Tête CTC :** **cosinus moyen entre lignes** des **matrices de projection CTC** des exécutions en aval **HF** vs **local** : **0,0321**, écart-type **0,0554** — le **classifieur** a appris des directions **différentes** alors que le **CER/WER agrégé** sur eng1 10 min pouvait **sembler identique**.

**Interprétation :** l’**espace des représentations** et le **classifieur** peuvent **tourner** fortement tandis que les métriques **mot** agrégées restent **proches** sous **supervision minimale**. C’est crucial si l’on compare des checkpoints sur **un seul CER sur la partition de développement**.

---

## Slide 19 — Conclusions à l’appui des données actuelles

Synthèse :

1. **HuBERT** reste, dans notre expérience, la baseline **la plus robuste** et **la plus reproductible** **sur les différents volets**.
2. **WavJEPA** **peut égaler HuBERT** en **monolingue eng1** lorsque le **checkpoint** est **bon** — HF ou local.
3. **JEPA minimal** est **clairement sous-entraîné** pour ce montage — ce **n’est pas** un substitut équitable pour « JEPA à grande échelle ».
4. Le **multilingue agrégé** apporte de **forts gains** avec des frontends **figés** par rapport à de petites portions monolingues.
5. **Question ouverte :** un pré-entraînement WavJEPA **from scratch soutenu**, **petit ou grand**, donne-t-il des gains en aval **stables** sur HuBERT dans des exécutions **propres** et **longues** ? Nous ne l’avons pas encore tranché.

---

## Slide 20 — Prochain bloc expérimental

Travaux prévus :

1. **Un run WavJEPA small from scratch propre** jusqu’à un **budget de pas cible** — pas de lignée de reprise ambiguë.
2. **A/B checkpoint monolingue complet :** **ancien** versus **nouveau** checkpoint, **même** recette ASR, **même** décodage, **même** politique de graines — pas de raccourcis de mise au point.
3. **Compléter les lignes manquantes :** **ASR+LID 1 h** et **LoRA 1 h** jusqu’au bout quand l’exploitation le permet.
4. **Publier un tableau canonique** avec **provenance** et **statuts** : **final**, **exploratoire** ou **interrompu** — pour que le récit soit **audit**.

---

## Slide 21 — Pont Janis + Bruny + Vadim (ASR, ABX, SER, profondeur)

Tu enchaînes le **fil rouge** du `report.tex` : **un même encodeur SSL figé** ne se résume pas à **un seul chiffre**.

**ASR (Janis)** : **CER/WER** sous budget MLSUPERB — mesure **lexicale** avec **tête CTC**.

**ABX** : sonde de **discriminabilité** « phonétique » dans l’espace des embeddings ; tu rappelles le cas **HuBERT** mieux que **WavJEPA HF** tout en **égalité** possible en **WER** sur eng1 10 min.

**SER (Bruny)** : sur **RAVDESS** et **IEMOCAP**, le **classement** change encore : **WavJEPA** est **très fort** en paralinguistique.

**Profondeur (Vadim + Bruny)** : l’**analyse par couches** côté **poids CTC** et côté **SER** dit **où** vit l’information ; la slide suivante sur le **proxy phonétique par couche** complète cette lecture sur la **tranche dev 311 énoncés**.

---

## Slide 22 — Carte « universalité » ASR vs SER

Tu montres la figure **`plots/13_asr_vs_ser_universality.png`** : en abscisse le **WER** eng1 10 min (plus c’est **bas**, mieux c’est pour l’ASR), en ordonnée l’**UAR** RAVDESS **full train** de Bruny (plus c’est **haut**, mieux c’est pour l’émotion).

**HuBERT** et **WavJEPA** sont **alignés** en WER mais **WavJEPA** monte en **SER**.

**JEPA minimal** est marqué **WER mauvais** ; la **SER** est **à lancer** avec le script **`ser_ravdess_frozen_frontend.py`** dès que le chemin **RAVDESS** est disponible — tu l’annonces comme **work in progress honnête**.

**wav2vec2** : tu rappelles qu’on n’a pas la ligne **ASR eng1** du même tableau dans le deck ; le point illustre surtout le **SER**.

---

## Slide 23 — Inversion des classements par tâche

Figure **`plots/14_ranking_inversion_across_tasks.png`** : une **heatmap de rangs** (1 = meilleur parmi les modèles qui ont un score sur cette colonne).

Colonnes : **ASR** via **1/WER**, **ABX** via **1/erreur** (chiffres du deck), **SER RAVDESS** et **SER IEMOCAP** (appendice du `report.tex`).

**Message oral :** les **couleurs qui bougent** d’une colonne à l’autre, c’est exactement **l’inversion de classement** : personne ne domine **toutes** les tâches avec la même hiérarchie.

---

## Slide 24 — Proxy phonétique par couche (pas fastabx officiel)

Tu affiches **`plots/15_layer_phonetic_proxy.png`** et tu **cadres** : ce n’est **pas** le **fastabx** paper-grade ; c’est un **proxy** de séparation **entre / intra** classes (première lettre du texte) après **moyennage temporel**, avec **features par couche** extraites via **`extract_*_per_layer_for_abx.py`** + **`layer_phonetic_proxy.py`**.

Tu relies **Vadim** (où la **tête CTC** met le poids) et **Bruny** (où l’**émotion** monte en **couches moyennes/tardives**) : ici tu montres une **courbe** qui parle de **structure phonétique** le long de la profondeur.

Tu mentionnes le **sanitizing NaN** sur certaines activations **WavJEPA** en hooks — transparence **méthodo**.

---

## Slide 25 — SER avec JEPA minimal (frontend figé)

**Objectif pédagogique :** tester si **JEPA minimal** est **aussi catastrophique** en **SER** qu’en **ASR**, avec le **même esprit** que Bruny (**backbone gelé**, petite tête).

**Commande type :** `uv run python scripts/ser_ravdess_frozen_frontend.py --ravdess_root ... --backend jepa_minimal`

**Statut :** **pas de corpus** dans le dépôt ; tu présentes ça comme **outil prêt** + **résultat à compléter** — ça valorise le **travail d’ingénierie** sans inventer de chiffre.

**Phrase de conclusion :** si SER reste **faible**, ça renforce « **pré-entraînement insuffisant** » ; si SER est **moins effondré** que l’ASR, ça **découple** pression **lexicale** vs **paralinguistique**.

---

## Slide 26 — Annexe : résultats SER (Bruny)

Cette annexe renvoie au travail du **collaborateur Bruny** sur la **reconnaissance des émotions dans la parole**, pas au tableau ASR MLSUPERB central.

Sur **RAVDESS** et en **transfert cross-corpus vers IEMOCAP**, **WavJEPA** a montré la **meilleure performance paralinguistique** parmi les modèles comparés dans cette étude.

**Analyse par couche :** les indices liés à l’**émotion** se concentrent plutôt dans les couches **moyennes à tardives** pour plusieurs architectures **SSL** — pas seulement la dernière couche.

**Leçon :** vision **dépendante de la tâche** — un modèle peut **mal performer en ASR** sur un benchmark et rester **fort en SER** sur un autre. Cette nuance compte quand on parle de « qualité de représentation » comme **un seul scalaire**.

---

## Slide 27 — Liens avec le cours MVA 2026 SNLP

Tu peux enchaîner **sans mentionner de dépôt GitHub** : la slide relie directement le contenu du cours *Speech and NLP* (MVA 2026, ENS/Inria) à ton travail, en **puces** :

- **Cours 5** (modèles acoustiques ASR, M. Poli) : tu illustres la même idée — **frontend gelé + CTC**, budgets **10 min / 1 h** façon MLSUPERB, c’est-à-dire **ASR sous très peu de parole étiquetée**.
- **Cours 6** (LM pour ASR, E. Dupoux) : le cours insiste sur les **LM au décodage** ; toi tu restes surtout **CTC** ; tu peux mentionner **ASR+LID** comme ouverture **multitâche** dans l’esprit « enrichir ce qu’on apprend au décodage ».
- **Cours 7** (speech foundation models, E. Dupoux) : c’est le **noyau** — **HuBERT vs WavJEPA**, **pré-entraînement SSL** puis **fine-tuning aval**, plus **ABX** pour la **discriminabilité** des représentations.
- **Évaluation** : le programme parle de **comparer performance humaine et machine** ; **CER/WER + ABX** montre que la « qualité » n’est pas **un seul** nombre.
- **Esprit projet** : **reproduire** une ligne solide + **prolonger** avec des questions nouvelles (HF vs local, multilingue, LoRA, robustesse pipeline / reprises).

---

## Slide 28 — Ce que j’ai appris : ABX, CER/WER, ASR, LID

**ABX :** le cours situe l’ABX comme sonde sur la **discriminabilité** (structure « phonétique » dans l’espace des représentations), pas comme un score de transcription. Le projet l’illustre **chiffré** : **HuBERT** fait mieux que **WavJEPA HF** en ABX tout en pouvant être **à égalité en CER/WER** sur eng1 10 min → **plusieurs métriques sont nécessaires**. Mettre en œuvre **fastabx** montre aussi qu’un résultat ABX **dépend d’une chaîne qui tourne** (ex. DTW / PyTorch).

**CER / WER :** ce sont les **mesures opérationnelles** de la qualité **ASR** sous **ton** protocole de décodage et **ton** budget d’étiquettes ; elles **réagissent** fortement au passage 10 min → 1 h, mais peuvent **diverger de la loss de pré-entraînement** SSL → même courbe de loss « propre » ne garantit pas des gains ASR proportionnels.

**ASR basse ressource :** le fil « modèles acoustiques + fondations » du cours correspond au **frontend gelé + petite tête CTC** ; le projet confirme que le **goulot** est surtout **l’entraînement ASR complet** (CTC, décodeur, I/O), et que le **multilingue agrégé** bat souvent le monolingue **minute** avec la même recette.

**LID :** l’**ASR+LID** conjoint à 10 min **dégrade** l’ASR pur par rapport à l’**ASR seul** au même budget → **compromis multitâche** net en ressource limitée ; ça relie le cours (objectifs annexes, LM/multitâche) à une **observation chiffrée** sur ta grille.

---

## Slide 29 — Questions

Je m’arrête ici et je prends vos **questions**. Si besoin, nous pouvons approfondir le **pré-entraînement**, les **bugs de reprise CTC**, le **protocole ABX** ou le **multilingue agrégé**. Merci.

---

## Veille de présentation (demain)

- **PDF :** recompiler une dernière fois `report_beamer.tex` si tu modifies quoi que ce soit ; vérifier que le **numéro de slide** dans ton lecteur PDF = **ordre du script** (pas de page de garde parasite).
- **Ne pas s’appuyer sur `report.md` pour la progression « globale »** : j’y ai ajouté un encart expliquant qu’il est **en retard** sur la piste Janis ; le **récit officiel**, c’est le deck + ce script.
- **Si on te coupe à 10 minutes :** enchaîner **1 → 4** (intro + question + périmètre), puis **7 → 11** (résultats + ABX + WavJEPA court), puis **12 → 14** (loss/CER + checklist + speech\_encoder **en 30 secondes**), **19 → 20** (conclusions + scope/« papier » + next **très court**), sauter ou survoler **15–18** (section technique temps/FLOPs/drift — à garder pour les questions), garder **une seule** slide parmi **21–25** (pont + figures) si tu veux l’effet « waw », sauter **26** (SER Bruny détaillé), **27–28** en **une minute**, questions.
- **Phrase de sécurité si on attaque sur les stats :** les **CER/WER** sur petites partitions sont des **instantanés de pipeline** ; l’argument fort reste la **comparaison à recette fixe** + la **dissonance ABX vs ASR**, pas un pourcentage « prouvé » au millième près.

---

## Conseil de timing (optionnel)

Si tu as **15 minutes** au total, vise à peu près : **1 min** page de titre, **2–3 min** objectifs + **question ABX vs ASR / JEPA** (slides 2–3) + **périmètre vs papier** (slide 4), **2 min** inventaire et ligne du temps, **3 min** tableaux de résultats et ABX, **~1 min** speech\_encoder vs ESPnet (slide 14), **3 min** loss vs CER et dérive des checkpoints, **2 min** conclusions et prochaines étapes, **~2–3 min** le **bloc pont + 3 figures** (slides **21–24**) si tu veux l’argument Vadim/Bruny, **~30 s** slide **25** (SER JEPA script), **~45 s–1 min** liens cours MVA (slide 27), **~1 min** apprentissages (slide 28), **1 min** questions. **29 slides** au total dans le PDF (après recompilation).
