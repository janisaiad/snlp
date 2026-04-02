ML-SUPERB et JEPA. Janis Vadim Bruny, 3 avril 2026.

Page de titre

Bonjour à toutes et à tous. Aujourd’hui, on vous présente notre projet autour de ML-SUPERB et des représentations de parole apprises en auto-supervision.

L’idée de départ est simple : aujourd’hui, les modèles SSL sont au cœur du traitement de la parole. On les préentraîne sur de très grandes quantités d’audio non transcrit pour apprendre des représentations riches et contextualisées, puis on les réutilise sur des tâches aval comme l’ASR, la diarisation ou la reconnaissance d’émotion.

Le cadre qui nous était proposé reprenait précisément la recette ML-SUPERB : prendre un modèle préentraîné comme HuBERT, wav2vec 2.0 ou un modèle proche, geler l’encodeur, puis n’entraîner qu’une petite tête CTC avec seulement dix minutes ou une heure de parole annotée par langue, donc dans un régime franchement bas ressources.

Les consignes proposaient aussi plusieurs extensions : comparer différents frontends, comparer mono et multilingue, regarder d’autres tâches comme l’identification de langue, et tester des variantes de fine-tuning léger comme LoRA ou d’autres méthodes PEFT. C’est exactement ce que nous avons essayé de couvrir ici : comparaison de représentations JEPA, WavJEPA et HuBERT sur une même recette downstream, expériences multilingues agrégées, branche ASR plus LID, adaptation LoRA, mesures ABX sur les représentations, et enfin analyse de la dérive entre checkpoint Hugging Face et checkpoint local. Ces trois slides résument donc ce qu’on a effectivement traité en complément du rapport principal report.tex.

Slide 1 — Problème et positionnement

Le protocole s’appelle ML-SUPERB. On prend un encodeur appris en auto-supervision, on le laisse gelé pendant l’entraînement de la tâche cible, et on n’apprend qu’une petite tête CTC par-dessus pour produire la transcription. Les budgets sont imposés par langue : environ dix minutes ou une heure de parole étiquetée, ce qui aligne tous les modèles sur les mêmes conditions de données rares.

La question centrale est de comparer des représentations JEPA ou WavJEPA à HuBERT avec strictement la même recette downstream — mêmes étapes, mêmes hyperparamètres côté tête, même pipeline — pour que les écarts reflètent l’encodeur, et non un autre choix d’implémentation.

Pour l’ASR, on utilise le CER et le WER : erreurs caractère et mot par rapport à la transcription de référence après décodage. Plus le pourcentage est bas, meilleure est la transcription. Les scores multilingues agrégés figurent sur la slide dédiée ; le détail par langue est dans report.tex.

Le multilingue porte sur un mélange anglais, allemand, français : ASR seul dix minutes ou une heure, ASR plus LID sur dix minutes, LoRA sur la branche ASR seul dix minutes.

Slide 2 — Représentations et benchmarks

Sur cette slide, il faut garder en tête que l’ABX et le CER ou le WER ne mesurent pas la même chose : l’ABX regarde la séparabilité phonétique dans l’espace latent, alors que le CER et le WER jugent l’alignement final avec le texte.

Le tableau multilingue et adaptation résume ASR seul dix minutes ou une heure, ASR plus LID dix minutes, et LoRA ASR seul dix minutes, avec CER et WER. La baseline est un HuBERT S3PRL gelé. Le message principal est simple : la LoRA n’apporte pas de gain net évident ici, alors que le passage de dix minutes à une heure produit un écart beaucoup plus marqué.

Le tableau ABX donne HuBERT vers 0,52, WavJEPA HF vers 0,59, avec un delta de plus 0,0679. Comme l’erreur ABX est plus basse pour HuBERT, ce delta positif se lit simplement comme un avantage pour HuBERT sur ce critère dans ce setup, sur le split dev_10min.

Slide 3 — Dérive des poids

Les durées affichées sont des heures GPU L4 mesurées sur la phase train dans ESPnet, et non le coût complet de toute la recette. On y voit le multilingue ASR dix minutes et une heure, le LID seul, ainsi que deux runs WavJEPA sur eng1 dix minutes, Hugging Face contre checkpoint local. Le wall-clock total du projet est plus grand.

On compare ensuite le même protocole avec un encodeur Hugging Face et un checkpoint local. Le cosinus moyen entre représentations HF et local est très proche de zéro, légèrement négatif, autour de moins 0,008, et le rapport des normes local sur HF vaut environ 1,59. Ici, une représentation par énoncé signifie simplement un vecteur qui résume chaque extrait audio, puis que l’on compare entre les deux runs pour le même fichier.

Le terme ctc_lo désigne, dans ESPnet, la couche linéaire de la tête CTC : c’est elle qui projette les sorties d’encodeur vers les logits sur les symboles du vocabulaire. Sa matrice de poids a donc une ligne par caractère ou jeton CTC. Quand le cosinus moyen entre ces lignes tombe autour de 0,032 entre le run Hugging Face et le run local, cela veut dire que les vecteurs classifieur par symbole peuvent beaucoup pivoter d’un entraînement à l’autre.

Au final, la perte SSL peut évoluer ou les représentations bouger sans que CER et WER suivent au millième près ; l’espace latent et la géométrie CTC peuvent tourner tout en gardant des scores ASR proches sous budget downstream serré.

Merci.
