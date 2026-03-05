# ML-SUPERB project: publishable directions and JEPA

**Context:** Frozen SSL (HuBERT, wav2vec 2.0, etc.) + CTC for ASR with 10 min / 1 h in low-resource languages, following ML-SUPERB (Shi et al., 2023). Code: [espnet/egs2/ml_superb](https://github.com/espnet/espnet/tree/master/egs2/ml_superb).  
**Supervisor:** Maxime Poli (maxime.poli@ens.psl.eu) — co-author of SpidR-Adapt (universal speech representation, few-shot adaptation; with Yann LeCun).

---

## Conceptual goal: transfer learning and universal concepts in latent space

**One way to state the goal:** Use **transfer learning across languages** to probe whether SSL representations capture **universal (language-independent) concepts** in latent space—i.e. structure that transfers with little or no language-specific training.

**How the project tests this:**
- **Frozen SSL + 10 min/1 h per language:** If the same frozen encoder does well on ASR (or LID) in many languages with only a small language-specific head trained on 10 min/1 h, then the **latent space already encodes reusable structure** (e.g. phonetic, phonological, or acoustic) that is **not purely language-dependent**. The head mainly learns a mapping from that shared space to orthography (or language ID).
- **Monolingual vs multilingual SSL:** Comparing HuBERT (trained on one language) vs XLSR (trained on many) asks: does **multilingual pre-training** yield a more **universal** latent space that transfers better to unseen or low-resource languages?
- **Which language benefits most from 10 min/1 h?** Languages that improve a lot with little data may be those whose structure is already well aligned with the SSL’s latent space (more “universal” in the representation); others may need more adaptation (more language-specific).
- **Frozen vs LoRA:** Keeping the SSL frozen stresses **reuse of existing latent structure**; LoRA allows the representation to adapt—so you can ask: when is the **frozen (universal) representation enough** vs when do we need **language-specific adaptation**?

**So yes:** The goal can be framed as **transfer learning across languages to see whether and where SSL latent spaces encode universal, non–language-dependent concepts**, and your experiments (which SSL, frozen vs LoRA, which languages, 10 min vs 1 h) are concrete ways to measure that.

---

## Core project spec: what it is and what to add

**As stated:** Use a **pretrained SSL** (HuBERT, wav2vec 2.0, etc.), **freeze** it, and implement **ASR with CTC** using only **10 min / 1 h** of speech in a **low-resource language**, following the **ML-SUPERB** procedure.

**What's strong about it:**
- **Clear and bounded:** One protocol (ML-SUPERB), one training setup (frozen SSL + CTC), two data regimes (10 min, 1 h). Easy to explain and reproduce.
- **Right benchmark:** ML-SUPERB is the standard for this exact setting; results are comparable to the literature and to future work.
- **Right question:** "How well can we do with 10 min/1 h and a frozen representation?" is practically and scientifically useful.
- **Feasible:** ESPnet recipe exists; you're not building the benchmark from scratch.

**What to add for a solid 2026 outcome:**
- **At least one non-frozen condition:** e.g. **frozen vs LoRA** (or adapter) with the same 10 min/1 h. The community (ML-SUPERB 2.0, 2025 challenge) now expects this; it also makes the story stronger ("when does freezing suffice?").
- **At least two SSLs:** e.g. HuBERT-base + XLSR-53 (one mono, one multi) so you get a **comparison** (which representation for low-resource), not just one model.
- **Optional:** One extra representation (JEPA or MMS) or one extra axis (mono vs multilingual finetuning) to turn it into a clear **contribution** rather than only a reproduction.

**Short take:** The core spec is good and sufficient for a **project report**. For a **publishable** result (workshop/short paper), keep the spec and add: **frozen vs LoRA** + **2+ SSLs** (and optionally one of: JEPA, MMS, or multilingual vs monolingual finetuning).

---

## SpidR-Adapt (supervisor’s work) and why JEPA is a good idea

**SpidR-Adapt: A Universal Speech Representation Model for Few-Shot Adaptation**  
Mahi Luthra, Jiayi Shen, **Maxime Poli**, Angelo Ortiz, Yosuke Higuchi, Youssef Benchekroun, Martin Gleize, Charles-Eric Saint-James, Dongyan Lin, Phillip Rust, Angel Villar, Surya Parimi, Vanessa Stark, Rashel Moritz, Juan Pino, **Yann LeCun**, Emmanuel Dupoux. arXiv 2025. [pdf](https://arxiv.org/abs/2512.21204) · [code](https://github.com/facebookresearch/spidr-adapt)

- **What it is:** Universal speech representation for **few-shot adaptation** to new languages with minimal data. Meta-learning (MAdaPT, FOBLO) over data-scarce scenarios; builds on SpidR (student–teacher SSL). **~100× data efficiency** vs standard training; strong results with **&lt;1 h** of target-language audio on ABX, sWUGGY, sBLIMP, tSC.
- **Relevance to your project:** Same **low-resource / few-shot** setting (10 min / 1 h). Your supervisor’s line is **efficient representations and adaptation**; ML-SUPERB is the right benchmark to compare representations (including alternatives to classic SSL). You could later compare **SpidR-Adapt** (or SpidR) in the ML-SUPERB protocol as another “efficient representation” baseline.

**Why JEPA is still a good idea (even with “only 2 JEPA papers”):**

- **Few papers = under-evaluated.** There are only a handful of JEPA-for-speech/audio papers (e.g. A-JEPA, Audio-JEPA, JEPA-as-tokenizer). So **benchmarking JEPA on ML-SUPERB** is *more* novel: you’re not repeating a crowded comparison, you’re **filling a gap** (first ML-SUPERB results for JEPA-style models).
- **Aligns with supervisor’s direction.** Maxime Poli works with Yann LeCun on **efficient speech representation** (SpidR-Adapt). LeCun is a central proponent of JEPA. Doing a **JEPA vs wav2vec2/HuBERT** comparison on ML-SUPERB fits the lab’s interest in **alternative representations** and low-resource behavior.
- **Bounded contribution.** You don’t design a new JEPA; you **evaluate** an existing one (e.g. Audio-JEPA) on the same 10 min/1 h protocol. That’s a clear, publishable “when does JEPA help in low-resource?” story.

**Short take:** “Only 2 JEPA papers” is a reason *to* do it: the space is open, your supervisor’s line (Poli + LeCun) is relevant, and a clean ML-SUPERB benchmark for JEPA is a solid contribution.

---

## What's the task? What's the ML-SUPERB procedure?

**Representations (taken as-is, frozen):**  
Pretrained SSL encoders (HuBERT, wav2vec 2.0, XLSR, etc.). You **do not** train them; you only use their **output features** as input to the downstream model.

**The new task (what you train):**  
You train **only the downstream** part to do **speech recognition** (and optionally **language identification**) on **10 min or 1 h** of labeled speech per setting. So the "new" task is:

- **Monolingual track:** **ASR** in one language — one (lang, dataset) pair for training; evaluate on all datasets of that language. Metric: **CER** (or **PER** for jpn/cmn).
- **Multilingual track (optional):** (1) **Multilingual ASR** — one model on 10 min or 1 h pooled over 143 languages; predict orthography. (2) **LID** — predict language ID. (3) **Joint ASR + LID** — both with lang ID at start of transcript. Metrics: **CER** for ASR, **accuracy** for LID.

So: **same (frozen) representations, new task = train a small head for ASR (and optionally LID) under 10 min/1 h.**

**ML-SUPERB procedure (step-by-step):**

1. **Data:** For each (language, corpus) use pre-defined **10-minute** and/or **1-hour** training subsets (and dev/test ~10 min each). Monolingual: pick one (lang, data) for training; multilingual: pool 10 min or 1 h over many languages.
2. **Features:** Run **frozen SSL** on audio → sequence of hidden states (one vector per frame). Optionally **weighted sum** over layers (learnable weights).
3. **Downstream model (the only part you train):** **Weighted sum** of SSL layers (learnable weights) → **Conv downsample** (sequence length ÷ 2) → **2-layer Transformer** (dim 256, FFN 1024, 8 heads, dropout 0.1) → **CTC loss** (predict character/phone sequence). **SpecAugment** on the (weighted) SSL representation.
4. **Training:** Adam, lr 0.0001, weight decay 1e-6, batch size 8, grad accumulation 4. **Monolingual:** 15,000 iterations. **Multilingual:** 300,000 (10 min) or 600,000 (1 h).
5. **Evaluation:** **CER** (or PER for jpn/cmn) for ASR; **accuracy** for LID. Report per language and/or average; optionally **SUPERBs** over all four tasks.

**In one line:** Take **frozen SSL features** → train a **small CTC head** (weighted sum + conv + 2-layer transformer) on **10 min or 1 h** of labeled speech → evaluate **CER** (and LID accuracy if you do that track).

---

## Research question: slightly harder, involves JEPA, fits Maxime Poli

**Poli's themes (from his work):** Few-shot / low-resource (SpidR-Adapt), **linguistic units** and **phonetic structure** (SpidR), **ABX discriminability** (fastabx, SpidR evaluation), spoken language modeling (sWUGGY, sBLIMP, tSC), **universal representations**, phoneme-level analysis (EMNLP 2024: phoneme classification for spoken LM), infant phonetic learning. He cares about **what representations capture** (phonetic vs lexical) and **efficient transfer**.

**Proposed research question:**  
**"Do JEPA and masked-prediction SSL (HuBERT/wav2vec2) learn similarly language-independent phonetic structure? Evidence from ABX discriminability and few-shot ASR transfer on ML-SUPERB. We also investigate when and how JEPA can do better** (e.g. frozen vs LoRA, layer selection, task/language) **— since JEPA often lags wav2vec2 on standard benchmarks."**

**Why compare JEPA vs the others? (conceptual hook)**  
JEPA is **guided toward predicting representations** in a joint embedding space, often in a framework oriented toward **action** and world model (LeCun: predict latent state, not pixels or labels). HuBERT and wav2vec2 are **guided toward reconstruction or discrete prediction** (masked tokens, quantized targets). So the **training objectives and inductive biases differ** — and the **representations can be different**. We don’t assume they are the same; we **investigate** whether they encode similar or different language-independent structure (phonetic, lexical, transfer under 10 min/1 h). That makes the comparison a real research question, not just “run another model on the same benchmark.”

**What you do (concretely):**
1. **Same representations:** Run **Audio-JEPA** (or another public JEPA) and **HuBERT-base** (and optionally XLSR-53) on the **same** set of ML-SUPERB languages.
2. **Two kinds of evaluation:**  
   - **ABX** (phonetic discriminability, zero-shot): use **fastabx** (Poli's library) or the standard ABX pipeline on (a subset of) ML-SUPERB languages. You get an ABX score per layer or per model.  
   - **Few-shot ASR:** standard ML-SUPERB protocol (10 min / 1 h, frozen encoder + CTC head). You get CER per language.
3. **The comparison:**  
   - Do JEPA and HuBERT **rank the same** on ABX vs on ASR? If JEPA is better on ABX but worse on ASR (or the opposite), that suggests they encode **different** aspects of "universal" structure (more phonetic vs more lexical).  
   - **Layer-wise:** Which layers of JEPA vs HuBERT best predict ABX vs ASR? Do the "best phonetic layer" and "best ASR layer" align? That ties to Poli's interest in **linguistic units** and **what transfers**.
4. **Optional:** Add **SpidR** or **SpidR-Adapt** as a third representation (same ABX + 10 min ASR) so the story is "JEPA vs masked-prediction vs SpidR-style" under the same protocol.

**Why it's slightly harder:** You combine **two evaluation frameworks** (ABX + ML-SUPERB ASR), possibly **layer-wise analysis**, and **two representation families** (JEPA vs HuBERT). It's not just "run one model on one task."

**Why Poli would find it great:**  
- Uses **ABX** (he wrote fastabx; SpidR/SpidR-Adapt are evaluated with ABX).  
- Directly about **phonetic vs lexical** and **what is universal** in the latent space — in line with SpidR ("linguistic units") and the conceptual goal of your project.  
- **JEPA** fits the lab's connection to LeCun and alternative representations.  
- **Few-shot ASR** is his bread and butter (SpidR-Adapt, 10 min/1 h).  
- One clear sentence for the abstract: "We compare JEPA and HuBERT on ABX and few-shot ASR; we find that … [phonetic and lexical transfer align / diverge in representation X]," which is a clean, publishable result.

**One-line formulation:** *Does JEPA learn more language-independent phonetic structure than HuBERT? We measure ABX (fastabx) and 10 min ASR (ML-SUPERB) on the same languages and compare layer-wise and model-wise.*

**JEPA often lags wav2vec2 on standard benchmarks — so we also ask: when can JEPA do better?**  
On many datasets (e.g. in the Audio-JEPA paper: ESC-50, FSD50k, GTZAN, LibriSpeech-MF, VoxCeleb, VoxLingua33, etc.), **Audio-JEPA does not outperform** (and sometimes underperforms) Wav2Vec2 and Data2Vec. So a second part of the research question is: **can we find conditions or adaptations under which JEPA does better** (or at least closes the gap) in the **low-resource setting**? Concretely: (1) **Frozen vs LoRA** — does adapting JEPA with LoRA on 10 min/1 h bring it closer to HuBERT/wav2vec2? (2) **Layer selection** — is there a layer or weighted combination where JEPA's representation is more transfer-friendly? (3) **Task and language** — does JEPA do relatively better on **ABX** (phonetic) than on **ASR** (lexical), or in certain language families? (4) **Data or curriculum** — would different pre-training or fine-tuning data help JEPA in this setting? So the **full research question** becomes: **compare JEPA vs masked-prediction SSL (HuBERT/wav2vec2) on ABX and few-shot ASR, and investigate when and how JEPA can do better** (conditions, adaptations, or analyses that close the gap).

**Another research question: is JEPA better in the spectrum?**  
A-JEPA (and Audio-JEPA) operate on **spectrograms** (e.g. mel), with latent-space prediction of masked **spectrogram patches**; HuBERT and wav2vec2 operate on **waveform** (or raw audio) and predict discrete or continuous **waveform-level** targets. The A-JEPA conclusion notes that "a straightforward application of JEPA yields remarkable outcomes for **audio spectrograms**" and that **time-frequency aware masking** (easy → hard) and **regularized masking** improve the learned representations. So a natural follow-up is: **is JEPA relatively stronger in the spectral (spectrogram) domain** than waveform SSL is in the waveform domain? Or: do JEPA-style representations (spectrogram-based) and wav2vec2-style (waveform-based) **complement each other** — e.g. JEPA better for certain phonetic or spectral structure, wav2vec2 for others? You could test this by: (1) comparing **A-JEPA / Audio-JEPA** (spectrogram) vs **HuBERT / wav2vec2** (waveform) on the **same** ABX and few-shot ASR setup; (2) asking whether JEPA **closes the gap or wins** on tasks that are more **spectral/phonetic** (e.g. ABX, PER) vs more **lexical** (ASR CER); (3) optionally, **fusing** spectrogram-JEPA and waveform-HuBERT features and seeing if the combination beats either alone. So: **"Is JEPA better in the spectrum?"** — i.e. does the spectrogram-based JEPA formulation give an advantage (or a different kind of universal structure) that we can detect in low-resource transfer? Add this as an optional or secondary research angle.

**Prior hypothesis: JEPA works better on phonetics (a priori).**  
A reasonable prior is that **JEPA performs relatively better on phonetic tasks** (e.g. ABX, PER) than on lexical tasks (e.g. ASR CER). Reasons: (1) JEPA predicts **continuous representations** in latent space rather than **discrete tokens** (phones/words), so its representation may align more with **phonetic** structure than with orthography. (2) Spectrogram-based JEPA (A-JEPA, Audio-JEPA) operates in the **spectral** domain, which is closer to classic phonetic features (formants, spectral envelope). (3) HuBERT/wav2vec2 are trained with **quantized** or **masked token** objectives that can push the representation toward **lexical** or subword units; JEPA has no such explicit lexical target. So we **hypothesize**: JEPA will close the gap (or win) more on **ABX / PER** than on **ASR CER**; the experiments (ABX + few-shot ASR on ML-SUPERB) test this. If the prior holds, the story is: "JEPA captures more language-independent **phonetic** structure; wav2vec2/HuBERT are stronger for **lexical** transfer with 10 min/1 h."

**Why can wav2vec2/HuBERT be strong on lexical tasks even though they are trained on waveform?**  
The **input** is waveform, but the **training objective** is what injects lexical/phonetic structure. (1) **HuBERT:** It predicts **cluster IDs** from an iterative clustering over the model’s own hidden states; those clusters converge toward **phone-like units**. So the model is trained to assign each frame to a discrete label that behaves like a phone — and phones map to words. (2) **Wav2vec2:** It uses a **quantized codebook** (Gumbel-softmax or similar); the model predicts **which codebook entry** (discrete “pseudo-phoneme”) for masked frames. So again the **target is discrete and reusable** across contexts. In both cases: **waveform in → encoder → predict discrete, phone-like targets**; the encoder is thus pushed to produce a latent space that **factors into discrete, lexical-friendly units**. Downstream, a small CTC head maps those units to orthography. So “trained on waveform” means “input is waveform”; the **supervision signal** (cluster IDs or codebook indices) is what makes the representation good for **lexical** ASR. JEPA, by contrast, predicts **continuous** latent vectors with no discrete target — hence the prior that JEPA might be more “phonetic” and less “lexical” than wav2vec2/HuBERT.

---

## Conclusion travaillée : question centrale — fusion JEPA + wav/HuBERT

**Synthèse après avoir bien bossé le sujet :**  
(1) Le papier ML-SUPERB (2023) est **vieux** — pour 2026 il faut s’appuyer sur ML-SUPERB 2.0, PEFT, et les références récentes. (2) Il y a des **liens avec Meta** via les papiers de **Maxime Poli** (SpidR, SpidR-Adapt, ABX/fastabx) — cadre naturel pour représentation universelle et few-shot. (3) Entre **sémantique** (ASR, CER) et **phonétique** (spectral, ABX, PER) il y a un **compromis à investiguer** : les deux types de structure ne sont pas forcément alignés dans une même représentation. (4) En SSL, **JEPA fonctionne bien en vidéo** mais **pas (encore) en audio** au niveau wav2vec2/HuBERT sur beaucoup de benchmarks (cf. Audio-JEPA, [arXiv:2507.02915](https://arxiv.org/abs/2507.02915) + code).

**La question profonde :**  
**Est-ce que JEPA (entraîné sur le spectre, biais inductif vers la phonétique) et wav2vec2/HuBERT (entraîné en waveform, biais vers le lexical/sémantique par le training) ont des features qui peuvent se fusionner pour faire un meilleur modèle global ?**

**Pourquoi c’est une excellente question :**  
- Elle suppose que les deux familles sont **complémentaires** (phonétique vs sémantique/lexical) plutôt que redondantes. Si JEPA apporte surtout de la structure phonétique/spectrale et wav/HuBERT de la structure lexicale, une **fusion** (concaténation, somme pondérée, gating appris, ou two-stream) pourrait donner un modèle qui bat chacun séparément sur **à la fois** ABX/PER et ASR/CER.  
- Elle est **testable** : on a déjà le protocole (ML-SUPERB, 10 min/1 h, ABX + ASR). On ajoute une condition **JEPA + HuBERT fusion** (par ex. concat des features par frame, ou weighted sum avec poids appris, ou petit module qui combine les deux) et on compare : JEPA seul, HuBERT seul, fusion — sur ABX, PER, CER.  
- Elle a un **lien clair avec Poli** (représentations universelles, ABX, compromis phonétique/lexical) et avec la littérature JEPA (vidéo vs audio, complémentarité des modalités).  
- Si la fusion bat les deux : **“Phonetic (JEPA) and lexical (wav2vec2/HuBERT) representations are complementary; fusing them yields a better global model.”** Si elle ne bat pas : on caractérise **où** chaque représentation est forte (ABX vs CER) et on documente le compromis. Les deux résultats sont publishables.

**Comment tester la fusion (concret) :**  
- **Early fusion :** Concaténer les features JEPA (spectrogramme) et HuBERT (waveform) au niveau frame (après alignement temporel si besoin) → une seule séquence de vecteurs → même downstream (weighted sum + conv + transformer + CTC).  
- **Late fusion / gating :** Deux branches (JEPA, HuBERT), chaque branche va jusqu’à un logit; combiner par somme pondérée ou par un petit MLP qui apprend les poids selon le contexte.  
- **Learned fusion :** Un petit module (attention ou MLP) qui prend [h_JEPA, h_HuBERT] et produit h_fused; entraîner le tout avec 10 min/1 h.  
- **Métriques :** ABX, PER (si dispo), CER — pour vérifier si la fusion améliore **les deux** (phonétique et lexical) ou seulement l’un.

**En une phrase :** Les features JEPA (biais phonétique/spectral) et wav/HuBERT (biais lexical/sémantique) sont-elles **complémentaires** au point qu’une fusion donne un meilleur modèle global ? C’est la question centrale à investiguer.

**Si je train JEPA avec CTC (comme pour wav), est-ce que je récupère de la sémantique ?**  
- **En sortie (ASR), oui :** Dès que tu mets une tête CTC sur JEPA et que tu l’entraînes sur des transcriptions (10 min/1 h), la **sortie** du système est sémantique (texte, mots). Le head apprend une application « représentation JEPA → caractères/phones » ; à l’inférence tu obtiens bien du lexical/sémantique.  
- **Dans la représentation JEPA elle-même, ça dépend :**  
  - **JEPA gelé (frozen) + CTC :** Tu n’entraînes que le head. La représentation JEPA reste celle du pré-entraînement (plutôt phonétique/spectral). C’est le **head** qui fait le travail de passer du « phonétique » au « lexical » ; tu « récupères » la sémantique **au niveau de la prédiction**, pas au niveau des features.  
  - **JEPA fine-tuné (ou LoRA) + CTC :** Les gradients de la loss CTC remontent dans JEPA. L’objectif CTC (prédire la bonne séquence de caractères) peut **pousser** les couches de JEPA vers des features plus utiles pour le lexical. Donc oui, en trainant JEPA avec CTC (en adaptant JEPA), tu peux **récupérer / injecter** une part de sémantique **dans** la représentation.  
- **En pratique :** Avec **frozen** JEPA + CTC tu as déjà de l’ASR (sémantique en sortie) ; la question est plutôt « est-ce que la représentation interne devient plus lexicale ? ». Avec **fine-tune / LoRA** JEPA + CTC, la représentation a une chance de devenir plus « sémantique » parce que la loss CTC la régularise dans ce sens. Pour comparer proprement JEPA vs wav/HuBERT, il faut clarifier : **frozen** (seul le head voit CTC) vs **adapté** (JEPA + head voient CTC) — dans le second cas tu « récupères » davantage de sémantique dans les features.

**Wav2vec2 est-il sémantique ou phonétique ? (et le rôle du fine-tuning)**  
Des travaux sur la **normalisation phonétique** dans wav2vec 2.0 (probing, fine-tuning) montrent que : (1) la **normalisation phonétique** n’est pas une étape explicite ; elle est **implicitement réalisée dans le modèle**. (2) **Fine-tuner** wav2vec 2.0 pour une tâche donnée **réalise une forme de normalisation** en **supprimant sélectivement l’information non pertinente** pour cette tâche. (3) Des modèles fine-tunés pour **plusieurs tâches** peuvent garder de l’info pour les deux sans dégrader les perfs ; supprimer l’info non pertinente n’est pas forcément nécessaire pour bien classifier. Donc **wav2vec2 n’est ni “purement” sémantique ni “purement” phonétique** : c’est **dépendant de la tâche** pour laquelle on le fine-tune. **Pre-trained** : les embeddings encodent plusieurs types d’info (phonétique, lexicale, speaker, etc.), avec des couches qui privilégient des aspects différents. **Fine-tuné pour ASR (CTC)** → la représentation est poussée vers ce qui sert le **lexical/sémantique** (normalisation “sémantique”). **Fine-tuné pour phonème / ton / speaker** → la représentation est poussée vers une **normalisation phonétique** (ou task-relevant). En résumé : wav2vec2 est **plastique** ; il devient plus “sémantique” ou plus “phonétique” selon la **tâche de fine-tuning**. D’où l’intérêt de comparer **frozen** vs **adapté** (LoRA/fine-tune) et **quelle tâche** (ASR vs ABX/PER) pour JEPA vs wav.

**Et HuBERT ?**  
Même idée que wav2vec2 : **ni purement sémantique ni purement phonétique**, ça dépend de la couche et de la tâche. La différence est que HuBERT est entraîné à prédire des **IDs de clusters** (clustering itératif sur ses propres hidden states), qui convergent vers des unités **de type phone**. Donc en **pré-entraînement**, HuBERT a déjà un biais un peu plus **phonétique** que wav2vec2 (cibles discrètes phone-like explicites). Une fois **fine-tuné pour ASR (CTC)**, les gradients poussent la représentation vers le lexical/sémantique, comme pour wav. Les études de probing montrent aussi que différentes couches encodent des aspects différents (acoustique vs phonétique vs lexical). En bref : HuBERT **pre-trained** = plutôt phonétique/lexical (unités discrètes) ; **fine-tuné pour ASR** = plus sémantique ; **fine-tuné pour phonème/PER** = normalisation phonétique. Donc **HuBERT et wav2vec2** sont tous les deux **plastiques** ; HuBERT part peut-être un peu plus “phonétique” à cause de l’objectif de clusters, mais le fine-tuning décide de ce qui domine.

---

## Clarification : phonétique vs lexical vs sémantique (pour ne pas mélanger)

- **Phonétique** = niveau des **sons** de la parole : phonèmes, traits acoustico-phonétiques, discriminabilité (même phone vs phone différent), indépendant du sens ou de l’orthographe. **Tâches / métriques :** ABX, PER (phoneme error rate), reconnaissance de ton, formants. **Représentation** = ce qui code “comment ça sonne”.
- **Lexical** = niveau des **mots / orthographe** : séquences de caractères ou de mots, vocabulaire, écriture. **Tâches / métriques :** ASR (transcription en texte), CER (character error rate), WER. **Représentation** = ce qui code “quelle séquence de mots (forme)”.
- **Sémantique** = niveau du **sens** : contenu, intention, paraphrase, similarité de sens. **Tâches :** compréhension (SLU), intention, résumé de sens, similarité sémantique. **Représentation** = ce qui code “ce que ça veut dire”.

**En pratique dans notre projet :**  
- On parle surtout de **phonétique** (ABX, PER) vs **lexical** (ASR, CER). L’ASR produit du **texte** (lexical) ; le “sens” n’est pas évalué directement.  
- Donc quand on dit “wav/HuBERT biais sémantique”, il vaut mieux dire **lexical** : bon pour la **transcription** (séquence de mots), pas forcément pour le sens profond.  
- **Résumé :**  
  - **Phonétique** = sons, phonèmes, ABX, PER.  
  - **Lexical** = mots, orthographe, ASR, CER.  
  - **Sémantique** = sens, intention, SLU, paraphrase.  
  JEPA (spectre, latent continu) → a priori plus **phonétique**. Wav/HuBERT (cibles discrètes, CTC → texte) → plus **lexical**. Le **sémantique** n’est pas au cœur du protocole ML-SUPERB (ASR + LID) ; on compare surtout **phonétique vs lexical**.

**Fine-tuner JEPA avec CTC (lexical) pour un modèle “complet” ?**  
L’idée : JEPA pré-entraîné apporte un biais **phonétique** (spectre, latent continu). Si on **fine-tune JEPA avec une CTC lexicale** (cibles = caractères/mots, comme en ASR), les gradients CTC poussent la représentation vers le **lexical**. On obtiendrait alors un **même** modèle avec : (1) base **phonétique** (pré-entraînement JEPA), (2) couche **lexicale** ajoutée par le fine-tuning CTC. En théorie ça pourrait donner un modèle **complet** (bon sur ABX/PER **et** sur ASR/CER) sans fusionner deux encodeurs. À tester : comparer **JEPA frozen + head CTC** vs **JEPA fine-tuné (ou LoRA) + CTC** sur ABX + ASR — est-ce que le JEPA fine-tuné reste bon en phonétique tout en devenant bon en lexical ? Si oui, “JEPA + CTC lexical” est un candidat pour un modèle unique phonétique+lexical.

---

## SUPERBs value (overall metric in ML-SUPERB)

**What it is:** SUPERBs is the aggregate score used in SUPERB/ML-SUPERB to summarize a model’s performance across all tasks and metrics in a single number (paper Sec. 2.4, Eq. 1).

**Definition:**  
Denote $s_{t,i}(u)$ = score of model $u$ on **task** $t$, **metric** $i$. $T$ = set of the four tasks (monolingual ASR, multilingual ASR, LID, joint ASR+LID). $I_t$ = set of metrics for task $t$. Then:

$$\text{SUPERBs}(u) = \frac{1000}{|T|} \sum_{t \in T} \frac{1}{|I_t|} \sum_{i \in I_t} \frac{s_{t,i}(u) - s_{t,i}(\text{FBANK})}{s_{t,i}(\text{SOTA}) - s_{t,i}(\text{FBANK})}$$

**Interpretation:**
- **FBANK** (baseline features): SUPERBs = **0**.
- **SOTA** (best model on each task in the benchmark): SUPERBs = **1000**.
- Any other model gets a value **between 0 and 1000** (or slightly above 1000 if it beats SOTA on some metrics). Higher = better.

So SUPERBs measures “how far the model is from baseline toward SOTA” on average across tasks and metrics, with task difficulty reflected because SOTA and FBANK are task-specific.

**In ML-SUPERB tables:** The last column is SUPERBs; e.g. XLSR-128 ≈ 947 (10 min) and 996 (1 h), FBANK = 0, HuBERT-base ≈ 832 (10 min) and 885 (1 h). You can report SUPERBs for your models to compare overall to the paper’s numbers.

---

## Which SSL models to use

**Recommended (good balance of signal vs compute):**

| SSL | Params | Pre-training | Why use it |
|-----|--------|--------------|------------|
| **HuBERT-base** | 95M | ~1k h, English | Strong monolingual baseline; ML-SUPERB shows HuBERT often beats wav2vec2 same size. Well supported in S3PRL/ESPnet. |
| **XLSR-53** | 317M | 56k h, 53 langs | Best multilingual in ML-SUPERB-style setups; 53 langs gives broad coverage. Large, so slower than base. |
| **wav2vec2-base** | 95M | ~1k h, English | Same size as HuBERT-base; lets you compare architecture (contrastive vs masked prediction) on equal footing. |

**If you can only run two:** **HuBERT-base** + **XLSR-53** — one monolingual (English) and one multilingual; you get “mono vs multi” and “which SSL for which language” in one go.

**If you want to keep compute low:** Use **base** models only (95M): **HuBERT-base**, **wav2vec2-base**, and **XLSR-53** (XLSR has no 95M variant, but one large is manageable). Or drop wav2vec2-base and run **HuBERT-base** + **XLSR-53** only.

**From ML-SUPERB paper:** XLSR-128 was best overall but 128-lang; XLSR-53 is easier to get and still very strong. HuBERT-large sometimes *worse* than HuBERT-base in multilingual low-resource; base models often generalize better. So prefer **base** for multi-SSL comparisons unless you explicitly want to study “large vs base.”

**Summary:** Default choice: **HuBERT-base**, **wav2vec2-base**, **XLSR-53** (three SSLs). Minimal: **HuBERT-base** + **XLSR-53** (two SSLs, mono + multi).

---

## ML-SUPERB (2023) paper: what’s still useful vs deprecated for a 2026 submission

**Short take:** The paper is still the right benchmark and protocol; the main gap for 2026 is that the community has moved to **ML-SUPERB 2.0** (fine-tuning + PEFT). So build on the 2023 setup but add at least one trainable condition so you’re not “frozen only.”

**Still useful (keep):**
- **10 min / 1 h** low-resource protocol — still the standard; not deprecated.
- **143 languages**, monolingual + multilingual tracks, **ASR + LID** — core design.
- **Frozen SSL + lightweight downstream** as *one* condition — still valid; use it as the efficient baseline.
- **FBANK baseline** — still the right reference.
- **Findings** (multilingual ≠ always better, HuBERT vs wav2vec2, base vs large) — still cited and relevant.
- **ESPnet recipe, data splits, corpora list** — use ML-SUPERB 2.0 / egs2 if available; otherwise 2023 setup is fine.

**Deprecated or insufficient for 2026:**
- **Frozen-only as the only setup:** ML-SUPERB 2.0 (Interspeech 2024) and the 2025 challenge add **fine-tuning** and **PEFT** (LoRA, adapters). A 2026 paper that *only* reports frozen SSL + CTC will look outdated unless you explicitly frame it as “when does frozen suffice?” or “efficient baseline.”
- **Ignoring PEFT:** For this benchmark family, 2026 reviewers will expect at least a **frozen vs LoRA** (or adapter) comparison.
- **Only “classic” SSLs:** ML-SUPERB 2.0 evaluates **supervised** models too (e.g. Whisper, OWSM). You don’t have to include them, but if you do “SSL only,” say so clearly (e.g. “we focus on SSL representations under low resource”).
- **Tiny downstream only:** 2.0 also uses larger downstream models. The 2-layer transformer + CTC is still fine as the *lightweight* option; just don’t claim it’s the only relevant setup.

**What to do for 2026:**
1. **Base protocol:** Use ML-SUPERB 1.0 (10 min/1 h, same tasks) or 2.0 if you use their splits/configs. Cite both 2023 and 2024 papers.
2. **Include at least one trainable condition:** e.g. **frozen vs LoRA** (or adapter) so your contribution is aligned with ML-SUPERB 2.0 and the 2025 challenge.
3. **Same languages and metrics** (CER, LID accuracy) — not deprecated.
4. **Optional:** One newer or supervised model (e.g. Whisper-small or OWSM) if you want to position against “full” 2.0; otherwise “SSL-only + PEFT” is a clear and sufficient scope.

**Summary:** The 2023 paper is not obsolete; its design and findings are still the foundation. For 2026, **add PEFT (or fine-tuning)** so your work is clearly “ML-SUPERB 1.0 protocol + frozen vs LoRA” or “ML-SUPERB 2.0–style,” not “frozen only” in isolation.

---

## MMS: Scaling Speech Technology to 1,000+ Languages (Pratap et al.)

**Reference:** Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, Michael Auli. “Scaling Speech Technology to 1,000+ Languages.” (MMS / Massively Multilingual Speech project; Meta.)

**Summary:** Expands speech technology from ~100 to **1,000+ languages**. Main pieces: (1) new data from **readings of publicly available religious texts** (e.g. Bible); (2) **self-supervised wav2vec 2.0** pre-training. Delivered: **wav2vec 2.0 pre-trained for 1,406 languages**, **one multilingual ASR model for 1,107 languages** (CTC + language-specific adapters, ~2M params per language), **TTS for 1,107 languages**, **LID for 4,017 languages**. On **FLEURS (54 languages)**, their multilingual ASR **more than halves WER vs Whisper** while using a **small fraction of labeled data**. Models on Hugging Face and fairseq.

**What’s strong:**
- **Scale:** 10–40× more languages than prior work; real step toward “speech for everyone.”
- **Data recipe:** Religious readings give consistent read speech and orthography in many languages; clever use of existing public data.
- **SSL + adapters:** Pre-train one wav2vec 2.0, then small per-language adapters; good fit for low-resource and on-demand loading.
- **Benchmark result:** Beating Whisper on FLEURS with less labeled data is a clear, reproducible result.
- **Open:** Pre-trained models and code available; easy to try in your own pipeline.

**Caveats / limitations:**
- **Domain:** Data is mostly read, religious text; may not match conversational or noisy speech; coverage and quality vary by language.
- **Evaluation:** FLEURS is 54 languages; performance on very low-resource or endangered languages is less documented.
- **Comparison set:** ML-SUPERB uses 143 languages and 10 min/1 h fine-tuning; MMS is “one big multilingual model + adapters” — different design (massively multi vs. per-language or small multi with limited data).
- **Fair comparison:** MMS is trained on much more (unlabeled) data per language in the religious corpus; so “MMS vs XLSR” is not apples-to-apples unless you control for data and protocol.

**Relation to your ML-SUPERB project:**
- **As an SSL to add:** MMS provides **wav2vec 2.0 checkpoints for 1,406 languages**. You could take a **single-language or small multi MMS checkpoint** and run it in the **ML-SUPERB protocol** (frozen or + LoRA, 10 min/1 h, same downstream). That would answer: “How does MMS representation compare to HuBERT/XLSR when adapted with 10 min/1 h?”
- **As a baseline model:** Their **released ASR model** (1,107 languages, CTC + adapters) is a strong multilingual baseline. You could compare **your 10 min/1 h ML-SUPERB setup** (e.g. frozen HuBERT + CTC) **vs zero-shot or light-adapted MMS** on the same ML-SUPERB languages/metrics.
- **As a data idea:** The “religious readings” recipe could inspire data collection for extra low-resource languages, but that’s a separate project from benchmarking on existing ML-SUPERB data.

**For a 2026 paper:** Including **MMS** (frozen or with LoRA) alongside HuBERT and XLSR in the ML-SUPERB protocol is **useful and timely**: it tests whether “massively multilingual” pre-training helps when you only have 10 min/1 h per language. One extra row (or column) in your main table is enough; no need to reproduce the full MMS training.

---

## What’s still relevant when big speech models (and evals) went brrrr

In a world where Whisper, OWSM, MMS, Kyutai-style open models, speech LMs, and benchmarks (MOS, quality evals, etc.) exploded, the project stays relevant by **not competing on raw SOTA** but on **questions big models don’t answer**.

**Still relevant:**

1. **“How much data do I need?”**  
   Big models are zero-shot or few-shot; they don’t tell you **data–performance curves** for a fixed representation (10 min vs 1 h vs 5 h). Your **data-scaling** experiments (Direction 1) answer that for SSL + CTC under a controlled protocol. Still useful for deployment and resource planning.

2. **When does frozen vs PEFT matter?**  
   With 10 min/1 h, **frozen SSL + CTC** vs **SSL + LoRA + CTC** is a real tradeoff (cost, overfitting, language coverage). Big models are usually used as-is or with heavy fine-tuning; your setting is the one where **efficient adaptation** is critical. Direction 3 stays central.

3. **Which representation for which language?**  
   “Monolingual vs multilingual SSL” and “best SSL per language/family” (Direction 4) are **scientific questions**: they don’t go away because Whisper exists. They inform when to use a big multi model vs a small mono one, and for which languages.

4. **Reproducible, bounded benchmark.**  
   ML-SUPERB (10 min/1 h, fixed splits, CER/LID) is a **controlled lab** for comparing representations and training strategies. Big-model evals are often noisier and harder to reproduce. Your contribution is **clean comparisons** on a fixed protocol, not “we beat X on a new test set.”

5. **Low-resource and endangered languages.**  
   Big models are still weak or missing for many languages and domains. Showing **what works with 10 min/1 h** (and which SSL, frozen vs LoRA, mono vs multi) is directly useful for under-resourced settings.

6. **Efficiency and cost.**  
   Frozen 95M SSL + small CTC head is **cheap** to train and run. That story (“when is this enough?”) remains relevant for on-device or low-budget deployment even if cloud APIs use huge models.

**Less relevant (or reframe):**

- **Beating Whisper/MMS in raw WER** on a random test set — not the goal; they have orders of magnitude more data. Your goal is **when and why** small-data adaptation works, not SOTA.
- **Introducing a new SSL** — not necessary; the value is in **systematic comparison** (frozen vs LoRA, mono vs multi, which SSL where) on an established benchmark.

**One-line take:** In the “speech models went brrrr” world, the project stays relevant by answering **when and how little data and which representation** are enough, and by doing it in a **reproducible, benchmarked** way that big-model papers often skip.

---

## Most publishable, simple but grounded directions

### 1. Data-scaling curves (strong baseline paper)

**Idea:** Same pipeline (frozen SSL + CTC), vary training size: e.g. 1 min, 5 min, 10 min, 30 min, 1 h (and optionally 3 h if data exists) for a small set of languages (e.g. 5–8: high/medium/low-resource).

**Output:** CER vs hours and “minimum data for acceptable CER” per language/model.

**Why it’s publishable:** Directly answers “how much data is enough?” for practitioners; no new architecture, just clear experiments and plots. Easy to replicate from the existing ESPnet recipe.

---

### 2. Layer / representation utility in low-resource

**Idea:** ML-SUPERB already uses a weighted sum of frozen layers. Run ablations: single-layer (or top-$k$ layers) vs full weighted sum for 10 min vs 1 h.

**Output:** “Best layer index (or layer range) for 10 min vs 1 h” and optionally per-language or per-language-family.

**Why it’s publishable:** Simple, no new training tricks; gives a clear rule of thumb (e.g. “middle layers dominate with 10 min”) and connects to representation analysis.

---

### 3. Frozen vs LoRA (or other PEFT) in 10 min / 1 h

**Idea:** Compare (a) fully frozen SSL + CTC head vs (b) SSL with LoRA (or adapter) + CTC, same data budget. Optionally: same for 2–3 SSLs (e.g. HuBERT-base, XLSR).

**Output:** Tables: CER and training cost for frozen vs PEFT at 10 min and 1 h.

**Why it’s publishable:** Aligns with ML-SUPERB 2.0 and the PEFT trend; low-resource is the natural setting where PEFT vs frozen matters most.

---

### 4. Systematic “which SSL for which language (family)”

**Idea:** Fix 8–12 languages (covering different families and scripts), run 3–4 SSLs (e.g. HuBERT-base, wav2vec2-base, XLSR-53, one multilingual large) with 10 min and 1 h.

**Output:** One main table: best model per language (or per family) and a short discussion (e.g. “multilingual SSL helps for X but not Y”).

**Why it’s publishable:** The ML-SUPERB paper already shows “multilingual doesn’t always win”; you make this systematic and reproducible on a fixed set of languages.

---

### 5. JEPA on ML-SUPERB (novel but bounded)

**Idea:** Run **A-JEPA** (or another public JEPA-style speech model) in the same ML-SUPERB protocol: frozen encoder + same downstream (e.g. CTC head), 10 min / 1 h for the same languages as above. Compare to HuBERT/wav2vec2.

**Output:** First ML-SUPERB results for JEPA-style SSL; “when does JEPA match or beat masked-prediction SSL in low-resource?”

**Why it’s publishable:** JEPA is under-evaluated on standard speech benchmarks; this is a clear, contained contribution (benchmarking, not designing a new JEPA).

---

## View on JEPA (and how it fits this project)

- **JEPA** (Joint-Embedding Predictive Architecture): the model predicts *representations* of masked/future regions in a joint embedding space instead of reconstructing input or predicting discrete labels. That can reduce collapse and improve generalization.

- **In speech:**
  - **A-JEPA** (“Joint-Embedding Predictive Architecture Can Listen”) already brings JEPA to audio/speech and does well on some tasks.
  - There are follow-ups (e.g. GMM-anchored JEPA, design studies for audio).

- **Why it’s interesting for this project:**
  - Different training objective (representation prediction vs masked token/label prediction) might behave differently in **low-resource** and **multilingual** settings.
  - A clean comparison “JEPA vs HuBERT/wav2vec2 on ML-SUPERB with 10 min/1 h” is **simple and novel**: no new method, just evaluation.

- **Caveat:** JEPA for speech is less standard than wav2vec2/HuBERT; you may need to integrate a third-party implementation (e.g. A-JEPA) into ESPnet/S3PRL and align data format and layer interface. So it’s a bit more engineering than 1–4, but still “bounded” and highly publishable.

---

## Two more JEPA papers (tokenizer + Audio-JEPA)

**1. JEPA as a Neural Tokenizer (Ioannides et al., incl. Yann LeCun)**  
“JEPA as a Neural Tokenizer: Learning Robust Speech Representations with Density Adaptive Attention.”

- **Setup:** Two-stage. (1) **JEPA + DAAM** (Density Adaptive Attention Mechanism): masked prediction in **latent space** only, no waveform reconstruction; Gaussian mixture–based density-adaptive gating in the encoder for adaptive temporal selection. (2) **Tokenization:** FSQ (Finite Scalar Quantization) + mixed-radix packing → **HiFi-GAN** decoder for waveform. Output: **2.5 Hz frame rate**, **47.5 tokens/sec**, reversible, language-model-friendly.
- **Take:** Pushes JEPA toward **neural codec / tokenizer** (discrete tokens + reconstruction), not just representation. DAAM adds structure (hierarchical, density-adaptive). For your project: if a **released checkpoint** exists, you could use the **encoder (stage 1)** as a frozen front-end in the ML-SUPERB protocol (like HuBERT) and compare; the tokenizer stage is optional unless you care about discrete units. Strong signal that JEPA is being taken seriously for speech (LeCun co-author).

**2. Audio-JEPA (Tuncay et al., IRIT-SAMoVA, QMUL)**  
“Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning.”

- **Setup:** JEPA for audio: **ViT** on **mel-spectrograms**, predict **latent of masked patches** (no raw reconstruction). Pre-trained on **AudioSet** (10 s clips, 32 kHz), random patch masking. Evaluated on **X-ARES** (speech, music, environmental).
- **Result:** **Comparable to wav2vec 2.0 and data2vec** with **&lt;1/5 of training data**, no hyperparameter tuning. Code/checkpoints on GitHub.
- **Take:** Simple, reproducible baseline: “JEPA on mel + ViT” already matches wav2vec2/data2vec with much less data. Good candidate to run on **ML-SUPERB** (frozen encoder → same CTC downstream, 10 min/1 h): if their checkpoint is on Hugging Face or GitHub, you get “Audio-JEPA vs HuBERT vs XLSR” with minimal new code. X-ARES is not ML-SUPERB, so **your contribution** = first ML-SUPERB results for this family.

**For your project:**  
- **Direction 5 (JEPA on ML-SUPERB)** can use **Audio-JEPA** (ViT + mel, easy to plug in) or **JEPA-as-tokenizer** encoder (if released) as the JEPA representative. Audio-JEPA is likely the lighter integration (standard mel + ViT; check compatibility with S3PRL/ESPnet input format).  
- Both papers support the story: **JEPA is a viable alternative to wav2vec2/HuBERT** for speech/audio; benchmarking them on **low-resource ML-SUPERB** is still an open and publishable gap.

---

## Recommendation

- **Highest payoff vs effort:** (1) data-scaling curves and (3) frozen vs LoRA.
- **Highest novelty with still manageable scope:** (5) JEPA on ML-SUPERB.
- Combining (1) + (3) or (3) + (5) would already make a solid workshop or conference short paper.

---

## Extension ideas (from project brief)

These are the suggested extensions from the project description; below is how they map to our directions and a short take.

| Brief idea | Maps to | Take |
|------------|--------|------|
| **Comparison with other monolingual or multilingual models** | Direction 4 (which SSL for which language) | Strong. Core of ML-SUPERB; run 3–4 SSLs (mono + multi) on the same 10 min/1 h setup and compare. |
| **Comparison of performance between finetuning languages** | Direction 1 (data curves) + Direction 4 | Strong. “Which language (or family) benefits most from 10 min vs 1 h?” is a clear, bounded question. |
| **Other tasks (phone recognition, LID, …)** | New axis | Good. LID is already in ML-SUPERB; phone recognition needs phone alignments/labels. Adding one extra task (e.g. LID only, or phone if you have a pipeline) keeps scope manageable. |
| **Other training procedure (LoRA or other PEFT as in ML-SUPERB 2.0)** | Direction 3 (frozen vs LoRA) | Strong. Directly in line with ML-SUPERB 2.0; low-resource is the right setting to compare frozen vs PEFT. |
| **Multilingual finetuning vs. monolingual finetuning** | New axis, combines with 3 and 4 | Strong. Train one model on 10 min of one language vs. 10 min pooled over several languages, same total budget; compare CER per language. Very publishable and answers a practical question. |

**Summary:** All five extensions are solid and fit the benchmark. The ones that combine well and stay simple: (1) **mono vs multi SSL** (Direction 4), (2) **frozen vs LoRA** (Direction 3), (3) **multilingual vs monolingual finetuning** (new, but same 10 min/1 h protocol). Doing one of these well is enough for a good report/paper; combining two (e.g. Direction 3 + multilingual vs monolingual finetuning) gives a stronger story.

---

## Other tasks (beyond ASR) that can lead to a publication

**LID (language identification) — explained**

- **What LID is:** Given a speech utterance, the model predicts **which language** it is (e.g. French, Swahili, Mandarin). So the output is a **language label**, not a transcript. Metric: **accuracy** (% of utterances correctly classified).
- **Already in ML-SUPERB:** The benchmark has a dedicated **LID track**: same 10 min/1 h training data (pooled over 143 languages), same pipeline (frozen SSL → small downstream head). You train the head to predict language ID instead of (or in addition to) text. No extra data or new setup.

**Three publishable angles (in plain language):**

1. **Frozen vs LoRA for LID**  
   You only change *how* you train: (a) **frozen SSL** + train a small classifier on top for LID; (b) **SSL + LoRA** + same classifier. Question: *When does adapting the SSL (LoRA) help LID under 10 min/1 h?* Maybe with very little data the frozen representation is enough; with 1 h, LoRA might help. You report LID accuracy for both and get a clear “when does adaptation pay off?” story.

2. **Which SSL is best for LID vs ASR?**  
   You run **the same SSLs** (e.g. HuBERT-base, XLSR-53) on **both tasks**: LID (predict language) and ASR (predict text), each with 10 min/1 h. Question: *Is the SSL that wins for ASR also the one that wins for LID?* Maybe multilingual SSL (XLSR) is better for LID but a monolingual one is better for ASR in a given language. One table: best SSL per task (and per language if you do mono ASR); that’s a concrete, publishable finding.

3. **LID-aware CTC (Wang et al., Interspeech 2025)**  
   Instead of training ASR and LID separately, you train **one model** that outputs **language ID first, then the transcript** (e.g. `[eng] hello world`). The model learns to “say” the language and then the words in one CTC sequence. Wang et al. (2nd place ML-SUPERB 2.0) show **+14% relative LID accuracy** and **−30% relative CER** vs a baseline without this trick. Your contribution: replicate their **LID-aware CTC** on ML-SUPERB 1.0 (or your chosen languages) and/or compare frozen vs LoRA in that setup. Clear, bounded, and aligned with the 2025 challenge.

**Verdict:** LID is the easiest “other task” (already in ML-SUPERB, same pipeline). Strongest publication angles: **frozen vs LoRA for LID**, **LID vs ASR: which SSL**, or **LID-aware CTC** (replicate or extend Wang et al.).

**Is LID “too simple”?**  
As a **standalone task** (train a classifier to predict language, report accuracy), LID is conceptually simple: it’s multi-class classification over ~143 languages. For a **thesis or paper**, that alone can feel light. It becomes **substantial** when you add a clear **scientific or practical question**: (1) **When does adaptation help?** (frozen vs LoRA for LID under 10 min/1 h.) (2) **Do the same representations work for LID and ASR?** (which SSL for LID vs ASR; ties to “universal latent space.”) (3) **Does joint modeling help?** (LID-aware CTC: one model for lang + transcript.) So: **LID + one of these angles** = enough for a solid contribution; **LID alone** (just accuracy numbers) = likely too thin. If you want a “heavier” task, **joint ASR+LID** (LID-aware CTC) or **phone recognition (PER)** or **ABX** are more involved; LID is then a **second metric** alongside ASR rather than the main story.

**Phone recognition (PER)**  
- **In ML-SUPERB:** PER is already used for Japanese and Mandarin (jpn, cmn). For other languages you need **phone-level labels** (force-aligned or from a G2P + lexicon).  
- **Publishable angle:** (1) **PER under 10 min/1 h for more languages** — add a few languages with phone alignments (e.g. from MFA, Kaldi, or existing phonemized corpora) and report PER; “how much do SSLs help phone recognition in low-resource?” (2) **PER vs CER:** correlation across languages and SSLs; when does good PER imply good CER?  
- **Verdict:** Strong if you have (or can get) phone alignments for 5–10 ML-SUPERB languages; otherwise skip.

**Phonetic discriminability (ABX)**  
- **What it is:** Triplet task — same phone (A, B same) vs different (X different); score = accuracy. Often **zero-shot** (no target-language labels). Used in Zero Resource Challenge, SSL evaluation (e.g. SpidR-Adapt uses ABX).  
- **Publishable angle:** **Same SSL, same 10 min — compare ASR (CER) vs ABX.** Do SSLs that are good for ASR with 10 min also good for ABX (and vice versa)? Use existing ABX setups (e.g. Zero Resource, S3PRL) on a subset of ML-SUPERB languages.  
- **Verdict:** Novel and publishable; no extra labeled data for ABX. Combines well with “which representation for low-resource?” (Direction 4).

**Joint ASR + LID (with LID token)**  
- **Already in ML-SUPERB:** One of the four tasks; model predicts [lang_id] + transcript.  
- **Publishable angle:** Focus the paper on **joint modeling**: LID-aware CTC or prefix tuning, frozen vs LoRA, which SSL. Compare **ASR-only vs joint ASR+LID** under 10 min/1 h — does joint training help or hurt CER/LID? Wang et al. (Interspeech 2025) is the reference; you can replicate on ML-SUPERB 1.0 or extend (e.g. more languages, different SSLs).  
- **Verdict:** Very publication-worthy; aligns with ML-SUPERB 2.0 challenge and recent LID-aware work.

**Summary — best “other task” for a publication:**  
1. **LID** (with frozen vs LoRA or LID-aware CTC) — no extra data, already in ML-SUPERB, clear story.  
2. **Joint ASR + LID** — focus on joint modeling and compare to ASR-only; strong fit with ML-SUPERB 2.0.  
3. **ABX** — zero-shot phonetic discriminability vs ASR with 10 min; novel comparison.  
4. **Phone recognition (PER)** — only if you can get phone alignments for several languages; then “PER under 10 min/1 h” or “PER vs CER” is publishable.

---

## What to do for a publishable, good contribution

**Minimal publishable contribution (pick one and do it well):**

- **Option A — Frozen vs PEFT (Direction 3):** Same 10 min / 1 h setup; compare **frozen SSL + CTC** vs **SSL + LoRA (or adapter) + CTC**. Report CER (and optionally LID) and training cost. Use 1–2 SSLs (e.g. HuBERT-base, XLSR-53) and 5–8 languages.
- **Option B — Multilingual vs monolingual finetuning:** Same total data (e.g. 10 min): (a) one model per language (monolingual), (b) one model on 10 min pooled over 3–5 languages (multilingual). Compare CER per language. Frozen SSL + CTC; 1–2 SSLs.
- **Option C — Which SSL for which language (Direction 4):** Fix 6–10 languages (different families); run **3–4 SSLs** (e.g. HuBERT-base, wav2vec2-base, XLSR-53) with 10 min and 1 h. One main table: best model per language/family + short analysis.

**Stronger contribution (combine two):**

- **Option A + B:** Frozen vs LoRA **and** for each, compare monolingual vs multilingual finetuning (same 10 min budget). One set of tables: 4 conditions × CER per language.
- **Option A + C:** Frozen vs LoRA for 2–3 SSLs across 6–8 languages; report which SSL and which training (frozen vs PEFT) wins per language.

**Concrete checklist:**

1. **Setup:** Clone ESPnet, run ML-SUPERB recipe (asr1) for at least one (lang, SSL, 10 min) to reproduce a baseline. Document environment and data splits.
2. **Experiments:** Run the chosen option (A, B, C or combination) with fixed seeds; log CER (and LID if relevant) per language and overall.
3. **Outputs:**  
   - Main table(s): CER (and LID) by condition (e.g. frozen vs LoRA, mono vs multi, or SSL × language).  
   - Optional: 1–2 figures (e.g. CER vs data size, or bar chart by language).
4. **Write-up:** Short paper (4–6 pages) or report: intro, method (ML-SUPERB protocol + your variant), results, discussion (which setting works when), conclusion.
5. **Reproducibility:** Share configs and scripts (or a fork/patch to ESPnet); mention data sources and splits so others can replicate.

**What makes it “good”:** Clear question, same protocol as ML-SUPERB, reproducible setup, honest reporting (including failures or ties), and a takeaway (e.g. “LoRA helps most for language X with 10 min; frozen is enough for Y with 1 h”).

---

## References (ML-SUPERB ecosystem, from Semantic Scholar)

1. **The ML-SUPERB 2.0 Challenge: Towards Inclusive ASR Benchmarking for All Language Varieties**  
   William Chen, Chutong Meng, +10 authors, Shinji Watanabe.  
   *Interspeech*, 17 August 2025.  
   **TLDR:** New test suite with 200+ languages, accents, and dialects to evaluate SOTA multilingual speech models; online evaluation server based on DynaBench for flexible model design.  
   [PDF](https://arxiv.org/abs/...)

2. **TalTech Systems for the Interspeech 2025 ML-SUPERB 2.0 Challenge**  
   Tanel Alumäe, Artem Fedorchenko.  
   *Interspeech*, 2 June 2025.  
   **TLDR:** LID and multilingual ASR system from Tallinn University of Technology; obtained top overall score in the challenge.  
   [PDF](https://arxiv.org/abs/...)

3. **Improving Multilingual Speech Models on ML-SUPERB 2.0: Fine-tuning with Data Augmentation and LID-Aware CTC**  
   Qingzheng Wang, Jiancheng Sun, Yifan Peng, Shinji Watanabe.  
   *Interspeech*, 30 May 2025.  
   **TLDR:** 14% relative improvement in LID accuracy and 30% relative CER reduction over baseline on ML-SUPERB 2.0; second place in Interspeech 2025 ML-SUPERB 2.0 Challenge.  
   [PDF](https://arxiv.org/abs/...)

4. **ML-SUPERB 2.0: Benchmarking Multilingual Speech Models Across Modeling Constraints, Languages, and Datasets**  
   Jiatong Shi, Shi Wang, +8 authors, Shinji Watanabe.  
   *Interspeech*, 12 June 2024.  
   **TLDR:** Benchmark for evaluating pre-trained SSL and supervised speech models across downstream models, fine-tuning setups, and efficient adaptation (e.g. PEFT); large performance gaps between languages and datasets.  
   [PDF](https://arxiv.org/abs/...)

5. **ML-SUPERB: Multilingual Speech Universal PERformance Benchmark**  
   Jiatong Shi, Dan Berrebbi, +8 authors, Shinji Watanabe.  
   *Interspeech*, 18 May 2023.  
   Extends SUPERB to multilingual: frozen SSL + lightweight downstream for ASR and LID over 143 languages (10 min / 1 h tracks).  
   [PDF](https://arxiv.org/abs/2305.10615)

6. **Findings of the 2023 ML-Superb Challenge: Pre-Training And Evaluation Over More Languages And Beyond**  
   Jiatong Shi, William Chen, +10 authors, Shinji Watanabe.  
   *ASRU (Automatic Speech Recognition & Understanding)*, 9 October 2023.  
   **TLDR:** 2023 ML-SUPERB Challenge expands SUPERB to self-supervised models for multilingual ASR and LID; benchmark covers 154 languages.  
   [PDF](https://arxiv.org/abs/...)
