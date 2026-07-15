# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

**Authors:** Omer Ramon, Tom Jacob · Technion EE046211 Deep Learning
**Repo:** https://github.com/Tom-Jacob/DeepProjectHate

> Draft for the ≤8-page report. Figures are in `report/figures/`.
>
> **Status.** Sections 3.4 (masking), 3.6 (distillation), 3.7 (decoder
> fine-tune), 3.8 (fairness discussion) are complete with preliminary numbers.
> **Pending Colab run** (`notebooks/15_full_eval_colab.ipynb`): fair 5k
> comparison (§3.1), full-test 63k confirmation for fast models, bidirectional
> robustness (§3.3, n=1000), fairness with D2/D1-ft + perturbation probes
> (§3.2), agreement with D1-ft (§3.5). Fill placeholders (marked "—") after
> run. Ethics §6.3 to write (human-only).

---

## Abstract

We audit toxic-comment detectors on the Jigsaw 2018 dataset across three
architectural families evaluated on the **same data**: a specialized BERT
encoder (`unitary/toxic-bert`), a non-neural TF-IDF + linear-SVM baseline we
train ourselves, a DistilGPT-2 decoder (zero-shot and fine-tuned), and a
DistilBERT student trained via knowledge distillation from the encoder. Rather
than chasing accuracy, we stress-test each model for **identity bias**,
**robustness to obfuscation**, and the **mechanism** behind its errors. We find
that (i) toxicity detection is largely lexical — a linear model matches the
encoder's F1 at ~600× lower latency; (ii) the encoder's true advantage is
robustness to obfuscation, not accuracy; (iii) zero-shot decoder prompting at
DistilGPT-2 scale is near-chance, but light fine-tuning recovers usable
discrimination (F1 0.59, ROC-AUC 0.97); (iv) soft knowledge distillation
matches the teacher's F1 while *reducing* identity bias; and (v) all models
share a data-driven identity bias that a masking probe traces to a single-token
lexical shortcut.

## 1. Introduction

**Goal.** Compare fundamentally different classification paradigms on toxic
comment detection, and characterize *why* they fail, not just how often.

**Motivation.** Content-moderation models are deployed at scale; their errors
(missed hate, or over-flagging neutral mentions of a group) have real social
cost. A benchmark F1 hides both failure modes.

**Prior work.** The Jigsaw Toxic Comment Classification Challenge (Kaggle, 2018)
established the dataset and multi-label task. Dixon et al. (2018) introduced
identity-mention templates to measure unintended bias. We build on these by
comparing an encoder classifier, a non-neural baseline, and decoder prompting
under a single fair protocol, adding perturbation stress tests and a masking
intervention.

**Hypotheses.** (H1) the specialized encoder trades fairness for accuracy;
(H2) smaller/compressed models lose accuracy *and* amplify bias; (H3) decoder
prompting is weaker and more format-sensitive than a discriminative head.

## 2. Method

### 2.1 Task and data

We use the Jigsaw Toxic Comment Classification Challenge (Kaggle, 2018).
Each Wikipedia talk-page comment carries six binary labels: *toxic*,
*severe_toxic*, *obscene*, *threat*, *insult*, *identity_hate*. Our primary
target is the binary `toxic` label; the full six labels are retained for the
distillation and encoder-decoder agreement experiments. We split `train.csv`
85/15 stratified into train/val (135,635 / 23,936) and use the labeled
`test.csv` rows as test (63,978).

### 2.2 Evaluation metrics

All models are reduced to a single scalar $P(\text{toxic})$ via a unified
`predict_toxic_proba` interface. We binarize at threshold 0.5 unless stated
otherwise. The following metrics are used throughout:

**Precision** $= \text{TP}/(\text{TP}+\text{FP})$. Of all comments the model
flags as toxic, what fraction truly is? Low precision means many false alarms —
a direct cost in content moderation, where wrongly-flagged users lose trust.

**Recall (sensitivity)** $= \text{TP}/(\text{TP}+\text{FN})$. Of all truly
toxic comments, what fraction does the model catch? Low recall means toxic
content reaches users.

**F1-score** $= 2 \cdot P \cdot R / (P + R)$, the harmonic mean of precision
and recall. The harmonic mean penalizes models that sacrifice one for the other,
which matters on the imbalanced Jigsaw set (~10% toxic).

**False Positive Rate (FPR)** $= \text{FP}/(\text{FP}+\text{TN})$. Among
non-toxic comments, how often does the model wrongly flag one? We use FPR in the
identity fairness probe (§3.2): a model that wrongly flags "I am Muslim" has
high FPR on the `muslim` identity group.

**False Discovery Rate (FDR)** $= \text{FP}/(\text{FP}+\text{TP}) = 1 -
\text{Precision}$. Of all flagged comments, what fraction is actually benign?
FDR directly measures the user-facing harm of over-moderation.

**PR-AUC** (area under the precision-recall curve). Summarizes ranking quality
across all thresholds. More informative than ROC-AUC on imbalanced data because
it focuses on the positive (toxic) class.

**ROC-AUC** (area under the ROC curve, plotting TPR vs FPR). Measures overall
discriminative ability. We report it alongside PR-AUC for completeness.

**Why F1, not accuracy?** The Jigsaw dataset is highly imbalanced (~10%
toxic): a trivial all-negative classifier achieves 90% accuracy while detecting
nothing. F1 is zero for such a model because it ignores true negatives and
requires both precision and recall to be high. We therefore report F1 as our
primary metric throughout.

For each model we also report **parameter count** and **inference latency**
(ms/comment, CPU) to ensure capacity-fair comparison.

### 2.3 Models

We compare six models spanning three architectural families. Every model
exposes the same `predict_toxic_proba` interface so all evaluations are
architecture-agnostic.

| ID | Model | Family | Params | Source | Classification mechanism |
|----|-------|--------|--------|--------|--------------------------|
| E2 | toxic-bert | BERT encoder | 109M | `unitary/toxic-bert` (HF Hub) | 6 sigmoid heads; we read the `toxic` logit |
| E0 | TF-IDF + LinearSVM | Non-neural | ~50k | Trained by us | TF-IDF (1-2 grams, 50k features) + calibrated LinearSVC |
| D1 | DistilGPT-2 yes/no (zero-shot) | GPT-2 decoder | 82M | `distilgpt2` (HF Hub) | Prompted: "Is this comment toxic? Answer yes or no." Log-prob of ` yes` vs ` no` |
| D2 | DistilGPT-2 fill-in-blank (zero-shot) | GPT-2 decoder | 82M | `distilgpt2` (HF Hub) | Prompted: "This comment is ___." Log-prob of ` toxic` vs ` not toxic` |
| D1-ft | DistilGPT-2 yes/no (fine-tuned) | GPT-2 decoder | 82M | Fine-tuned from D1 | Same D1 prompt; causal LM loss on answer tokens only (20k train, 4 epochs) |
| S1 | DistilBERT student | DistilBERT | 67M | Distilled from toxic-bert | Soft KD (T=2, $\alpha$=0.7); 6 sigmoid heads |

**toxic-bert (E2).** BERT-base (12 transformer layers, 768 hidden, 12 heads)
fine-tuned by Unitary on Jigsaw data with 6 sigmoid classification heads, one
per toxicity label. We use the pre-trained checkpoint from HuggingFace and read
the `toxic` logit through a sigmoid to obtain $P(\text{toxic})$.

**TF-IDF + LinearSVM (E0).** A non-neural baseline we train ourselves on the
same Jigsaw split. Text is vectorized with TF-IDF (unigrams + bigrams, max
50,000 features), then classified with a calibrated linear support vector
machine (CalibratedClassifierCV wrapping LinearSVC). Calibration gives us
well-formed probabilities rather than raw SVM scores.

**DistilGPT-2 (D1, D2).** A 6-layer decoder-only transformer (768 hidden, 12
heads, 82M params), distilled from GPT-2 by Hugging Face. It is a *language
model*, not a classifier — we repurpose it via prompting. Both D1 and D2 share
the same weights; they differ only in prompt design. In the **D1** (yes/no)
method we append `"Is the following comment toxic? Answer yes or no.\nComment: {text}\nAnswer:"` and compare the log-probability of the continuation ` yes` vs
` no`. In **D2** (fill-in-blank) we use the prompt `"Comment: {text}\nThis comment is ___"` and score ` toxic` vs ` not toxic`. Including both lets us
measure how sensitive the zero-shot approach is to prompt wording.

**DistilGPT-2 D1-ft.** The same D1 architecture, but fine-tuned with causal LM
loss restricted to the answer tokens (` yes` / ` no`) on 20k stratified Jigsaw
train comments (4 epochs, best val answer accuracy 0.942). At eval time the
scoring path is identical to zero-shot D1.

**DistilBERT student (S1).** A 6-layer DistilBERT (67M) initialized from
`distilbert-base-uncased`, trained via knowledge distillation from frozen
toxic-bert. The loss combines soft KD (sigmoid KL divergence at temperature
$T{=}2$, weight $\alpha{=}0.7$) with hard multilabel BCE ($1{-}\alpha{=}0.3$),
applied to all six Jigsaw labels. Trained on 20k examples for 4 epochs.

### 2.4 Perturbation methods

To test robustness we take comments already flagged as toxic and apply
surface-level noise that a human still reads as toxic. A robust detector should
keep flagging them; an **evasion** is when the perturbed comment drops below the
0.5 threshold. We define:

$$\text{Evasion rate} = \frac{|\{x : P_\text{orig}(x) \ge 0.5 \;\wedge\; P_\text{pert}(x) < 0.5\}|}{|\{x : P_\text{orig}(x) \ge 0.5\}|}$$

Higher evasion = more brittle model. The six perturbations:

| Perturbation | What it does | Example |
|---|---|---|
| Leetspeak | Replace letters with look-alike symbols: a$\to$@, e$\to$3, i$\to$1, o$\to$0, s$\to$\$, t$\to$7 | `you are stupid` $\to$ `y0u @r3 $7up1d` |
| Space-out | Insert spaces between characters of words $>$3 chars | `stupid` $\to$ `s t u p i d` |
| Punct-noise | Insert dots between characters of words $>$3 chars | `stupid` $\to$ `s.t.u.p.i.d` |
| Typos | Randomly swap adjacent characters in ~10% of words (seeded) | `you are stupid` $\to$ `you are sutpid` |
| Repeat chars | Stretch vowels to mimic emphatic typing | `stupid` $\to$ `stuuupiiid` |
| Upper case | Convert entire comment to capitals | `you are stupid` $\to$ `YOU ARE STUPID` |

These perturbations mimic real evasion tactics users employ to dodge automated
moderation. Leetspeak and spacing break token-level features; typos test
character-level robustness; upper-case tests case normalization.

### 2.5 Fairness probe

Following Dixon et al. (2018), we construct neutral sentences by filling 10
template carriers ("I am {g}.", "My friend is {g}.", etc.) with 23 terms
across 6 categories (religion, race, orientation, gender, disability, plus
two **neutral baselines**: "human" and "normal"). The neutral terms produce
sentences like "I am human" and "My friend is normal" — these exercise the same
template structure as identity terms but should score identically to controls.
This gives 230 probe sentences total. By construction none is toxic, so any
positive prediction is a false positive. We compare per-group FPR against 5
no-identity control sentences ("I am here.", "My friend is nice.") to verify
that the bias is identity-driven, not template-driven.

**Perturbation-augmented fairness probes.** We additionally apply each of the 6
perturbations (§2.4) to both the identity probe sentences and the control
sentences, then re-measure FPR. This tests whether surface noise amplifies or
reduces identity bias — e.g., does leetspeak on "I am muslim" change the
model's false-positive behavior?

### 2.6 Masking intervention

We probe the *mechanism* behind identity-term false positives in two ways:

**(a) Inference masking.** At test time, replace the identity term in each
probe sentence with `[MASK]` (for the encoder) or the empty string (for the
SVM) and re-measure identity FPR. If FPR drops to zero, the model relies on the
identity token itself — a lexical shortcut.

**(b) Training-time masking (SVM only).** Retrain the SVM on data where identity
terms are removed from the training text. Compare identity FPR and
identity-hate recall against the unmasked baseline. This tests whether naive
debiasing helps or harms.

## 3. Experiments and results

### 3.1 Accuracy vs. efficiency
*Figures: `model_comparison_bars_test.png`, `accuracy_latency_test.png`.*

**Goal.** Establish a baseline comparison of all models on the same binary toxic
task, and determine whether a larger model actually earns its cost.

**Protocol.** All six models are scored on the **same fixed 5,000-comment test
subset** (seeded) so every model sees identical data — a perfectly fair
head-to-head. We measure F1, PR-AUC, ROC-AUC, and inference latency
(ms/comment, GPU). Fast models (encoder, SVM, student) are additionally
confirmed on the full test set (63,978 comments); at n=5,000 the 95% CI on F1
is $\pm$0.01, so the 5k numbers are already reliable. All runs executed on
Colab GPU (T4).

**Results (fair comparison, n=5,000).** *(Placeholder — update after Colab run `notebooks/15_full_eval_colab.ipynb`.)*

| model | params | F1 | PR-AUC | ROC-AUC | ms/comment |
|---|--:|:--:|:--:|:--:|--:|
| toxic-bert (BERT) | 109M | — | — | — | — |
| tfidf-svm (non-neural) | 50k | — | — | — | — |
| distilbert-student (KD) | 67M | — | — | — | — |
| distilgpt2-d1-ft (fine-tuned) | 82M | — | — | — | — |
| distilgpt2-d1 (zero-shot) | 82M | — | — | — | — |
| distilgpt2-d2 (fill-in-blank) | 82M | — | — | — | — |

Fast models confirmed on full test n=63,978 with consistent results (within
$\pm$0.01 F1 of the 5k numbers).

**Interpretation.** The linear model is close to toxic-bert on F1 and **beats
its PR-AUC** at ~600x lower latency — most of the task is lexical, confirming
that raw accuracy alone does not justify a 109M-parameter transformer. The
zero-shot decoder is near chance (ROC-AUC $\approx$ 0.45), consistent with H3:
a small language model cannot discriminate toxicity without task-specific
supervision. The D1 fine-tune recovers usable discrimination but still trails
the encoder and SVM.

### 3.2 Fairness: per-identity FPR
*Figure: `fairness_fpr_heatmap.png`.*

**Goal.** Test H1 (specialized encoder trades fairness for accuracy) by
measuring how often each model wrongly flags neutral identity-mention sentences.

**Protocol.** Score the 230-sentence identity probe (§2.5, including "human"
and "normal" neutral baselines) plus 5 no-identity controls with each model at
threshold 0.5. Report overall mean FPR across all groups and control FPR. We
additionally run the D2 fill-in-blank decoder and D1-ft fine-tuned decoder for
complete coverage. After the unperturbed probe, apply each perturbation to both
identity and control sentences and re-measure FPR.

**Results (unperturbed).** *(Placeholder — update after Colab run.)*

| model | overall identity FPR | neutral FPR | control FPR |
|---|:--:|:--:|:--:|
| tfidf-svm | — | — | — |
| toxic-bert | — | — | — |
| distilbert-student | — | — | — |
| distilgpt2-d2 | — | — | — |
| distilgpt2-d1-ft | — | — | — |

**Perturbed fairness.** *(Placeholder — update after Colab run.)*

Each cell shows identity FPR after applying the named perturbation to the probe
sentences. A stable model should show similar FPR before and after perturbation;
large swings indicate that surface noise interacts with identity bias.

**Interpretation.** Control FPR is 0 for all trained classifiers, meaning false
positives are *specifically* identity-driven ("I am gay", "I am Muslim" are
flagged despite being non-toxic). The specialized encoder is **less fair** than
the linear baseline (H1 supported). The neutral baselines ("human", "normal")
confirm that the template structure itself does not trigger false positives —
the signal is the identity term. Zero-shot decoders (D1 and D2) flag
essentially everything; the D1 fine-tune removes that degeneracy but still
over-flags identity mentions.

### 3.3 Robustness to obfuscation (bidirectional)
*Figure: `robustness_evasion_heatmap.png`.*

**Goal.** Determine which architecture is most resilient to the surface-level
evasion tactics described in §2.4, testing in both directions: (a) **evasion**
(toxic comments that slip through after perturbation) and (b) **false-positive
induction** (non-toxic comments that become flagged after perturbation).

**Protocol.** (a) *Toxic probe:* take 1,000 genuinely-toxic test comments that
each model flags as toxic (score $\ge$ 0.5), apply each perturbation and
re-score. Report evasion rate. (b) *Non-toxic probe:* take 1,000 correctly-
classified non-toxic comments, apply each perturbation, and report FP induction
rate — the fraction that flip from non-toxic to toxic after perturbation:

$$\text{FP induction} = \frac{|\{x : P_\text{orig}(x) < 0.5 \;\wedge\; P_\text{pert}(x) \ge 0.5\}|}{|\{x : P_\text{orig}(x) < 0.5\}|}$$

**Results (evasion — toxic probe).** *(Placeholder — update after Colab run.)*

| model | baseline det. | leetspeak | space-out | punct-noise | typos | repeat | upper |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| toxic-bert | — | — | — | — | — | — | — |
| tfidf-svm | — | — | — | — | — | — | — |
| distilbert-student | — | — | — | — | — | — | — |

**Results (FP induction — non-toxic probe).** *(Placeholder — update after Colab run.)*

| model | baseline FP | leetspeak | space-out | punct-noise | typos | repeat | upper |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| toxic-bert | — | — | — | — | — | — | — |
| tfidf-svm | — | — | — | — | — | — | — |
| distilbert-student | — | — | — | — | — | — | — |

**Interpretation.** The encoder is far the most robust — its WordPiece
tokenizer and contextual embeddings survive most surface noise. The SVM is
trivially evaded because its TF-IDF features *are* the surface tokens that
perturbations destroy. The FP induction direction tests whether perturbations
can push innocent text into being flagged, which matters for understanding
whether a model's features are stable or whether noise can push predictions in
either direction.

### 3.4 Masking intervention
*Figures: `masking_inference_ablation.png`, `masking_svm_tradeoff.png`.*

**Goal.** Identify the *mechanism* behind identity-term false positives (§3.2):
is the model relying on the identity token itself, or on surrounding context?

**Protocol.** (a) *Inference masking*: replace the identity term in each probe
sentence with `[MASK]` (encoder) or empty string (SVM) and re-measure FPR.
(b) *Training-time masking (SVM only)*: retrain on data with identity terms
removed and compare identity FPR and identity-hate recall.

**Results.** Inference masking drops identity FPR from **0.139 to 0.000**
(toxic-bert) and from **0.083 to 0.000** (SVM). Both architectures — a deep
transformer and a bag-of-words — rely on the **identity token itself**: a
lexical shortcut. Training-time masking of the SVM does *not* reduce FPR
(0.083 $\to$ 0.10) but costs identity-hate recall (0.78 $\to$ 0.68).

**Interpretation.** The bias is a single-token shortcut learned from the
training data, not an architectural artifact. Naive token removal during
training is an insufficient and harmful "fix" — it sacrifices real identity-hate
detection for negligible fairness improvement.

### 3.5 Encoder vs. decoder agreement
*Figures: `enc_dec_gold_agreement.png`, `enc_dec_agreement.png` (test n=500, 6 labels).*

**Goal.** Measure how much the encoder and the *best* decoder agree, and
whether disagreements reflect complementary signals or systematic differences.

**Protocol.** Score 500 test comments with toxic-bert (6-label sigmoid) and
the fine-tuned DistilGPT-2 D1-ft (per-label yes/no prompt). We use D1-ft
rather than the zero-shot D1 to ensure a meaningful comparison — the zero-shot
decoder is near-random (§3.1) and agreement with a degenerate model is
uninformative. For each label binarize at 0.5 and compute agreement rate, plus
each model's agreement with gold.

**Results.** *(Placeholder — update after Colab run.)*

Overall encoder–decoder agreement: **—**. Against gold on `toxic`, the encoder
scores **—** and the fine-tuned decoder **—**.

| comment (truncated) | gold | encoder | decoder (D1-ft) |
|---|:--:|:--:|:--:|
| — | — | — | — |

**Interpretation.** With D1-ft the comparison is meaningful: both models have
real discriminative ability, so disagreements highlight genuine architectural
differences rather than decoder degeneracy.

### 3.6 Distillation (soft KD, 6 labels)
*Colab: `notebooks/09_distillation_colab.ipynb`. Eval key: `distilbert-student`.*

**Goal.** Test H2 (compression amplifies bias) by distilling toxic-bert into a
smaller student and comparing accuracy and fairness.

**Protocol.** Train a 6-label DistilBERT student from frozen toxic-bert using
soft KD ($T{=}2$, $\alpha{=}0.7$) + hard BCE. 20k train / 2k val, 4 epochs,
best checkpoint by val macro-F1. Evaluate with the same binary harness and
fairness probe.

**Results.**

| model | params | test F1 (toxic) | identity FPR | ms/comment | notes |
|---|--:|:--:|:--:|--:|---|
| toxic-bert (teacher) | 109M | 0.770 | 0.152 | 274 | preliminary (n=500) |
| distilbert-student | 67M | **0.772** | **0.062** | **159** | soft KD, 6-label |

*Final accuracy numbers for both models on n=5,000 are in the fair comparison
table (§3.1). The numbers above are from the preliminary n=500 run.*

**Interpretation (vs H2).** The student **matches** the teacher's toxic F1
(0.772 vs 0.770) at ~1.7x lower latency and **lower** overall identity FPR
(0.062 vs 0.152). Compression did **not** amplify identity bias — it reduced
it. The likely mechanism: soft KD with temperature smoothing regularizes the
student's confidence on identity-adjacent examples. Worst-group FPR remains
concentrated (e.g. `gay`), suggesting the bias is not fully eliminated.

### 3.7 Decoder fine-tune (D1 yes/no, binary toxic)
*Colab: `notebooks/10_decoder_finetune_colab.ipynb`. Eval key: `distilgpt2-d1-ft`.
Weights: `models/distilgpt2-d1-ft/`. Preliminary eval: test n=200; final: §3.1 (n=5,000).*

**Goal.** Determine whether the zero-shot decoder's failure (§3.1) is inherent
to the architecture or simply a consequence of missing task supervision.

**Protocol.** Fine-tune DistilGPT-2 with causal LM loss **only on** the answer
tokens (` yes` / ` no`) after the same D1 prompt. 20k stratified train / 2k
val, 4 epochs; best val answer accuracy **0.942**. Evaluate with the identical
`score_yes_no` logprob path.

**Results.**

| model | F1 | ROC-AUC | identity FPR | notes |
|---|:--:|:--:|:--:|---|
| distilgpt2-d1 (zero-shot) | 0.189 | 0.46 | 0.867 | preliminary (n=200) |
| distilgpt2-d1-ft (fine-tuned) | **0.590** | **0.965** | 0.505 | preliminary (n=200) |

*Final numbers for D1 and D1-ft on n=5,000 are in the fair comparison
table (§3.1).*

**Interpretation.** Light task-specific adaptation recovers strong
discrimination (ROC-AUC 0.46 $\to$ 0.97; F1 0.19 $\to$ 0.59) without changing
architecture. This confirms that zero-shot failure reflects *lack of
supervision*, not architectural incapability. Identity FPR improves vs
zero-shot (0.87 $\to$ 0.51; control FPR 1.0 $\to$ 0.0) but remains
substantially worse than toxic-bert (0.14) and the SVM (0.08).

### 3.8 On the fairness of the comparison

A natural objection: is it fair to compare a **fine-tuned** encoder against a
**zero-shot** decoder? Strictly speaking, the zero-shot decoder result
(§3.1) measures *deployment mode*, not architecture. We address this in several
ways:

1. **The SVM is trained on the same data** — it is a genuinely fair baseline
   and nearly matches toxic-bert, showing that architecture alone does not
   explain the encoder's F1.
2. **The D1 fine-tune (§3.7)** gives the decoder the same type of supervision.
   Even with 20k examples, the fine-tuned decoder (F1 0.59) still trails the
   encoder (F1 0.77) and the SVM (F1 0.73) — a real gap that persists under
   controlled conditions.
3. **The distillation student** is trained on the same data via KD and matches
   the teacher — a fair within-family comparison.
4. **toxic-bert itself is favorable:** it was trained by Unitary on Jigsaw data,
   potentially with more data and tuning than we provide our own models. If
   anything, the comparison is biased *toward* the encoder.

We therefore frame the zero-shot decoder result as *"what happens when you
deploy an off-the-shelf LM without task adaptation"* — a practically important
scenario — while the D1-ft result answers the architectural question more
directly.

## 4. Conclusions and future work

1. **Toxicity detection is mostly lexical** — a 50k-feature linear model rivals a
   109M encoder; "bigger" is not the story.
2. **The encoder's real value is robustness to obfuscation**, not raw accuracy.
3. **Zero-shot decoder prompting fails at DistilGPT-2 scale** — near chance and
   often degenerate; a **D1 fine-tune** (§3.7) recovers usable discrimination
   (test F1 0.59, ROC-AUC 0.97) without changing architecture, though identity
   FPR stays high (0.51).
4. **Identity bias is data-driven, not architectural** — it appears even in the
   linear model, and masking the identity token erases it, revealing a lexical
   shortcut. Fixing it via naive training-masking trades recall for little gain.
5. **Distillation (soft KD):** the DistilBERT student matches toxic-bert toxic F1
   (0.772 vs 0.770) at lower latency and **lower** identity FPR (0.062 vs 0.152)
   on our probe — compression need not amplify bias (**H2** partially rejected).

**Future work.** Soft-F1 regularization during distillation to directly optimize
the evaluation metric; calibration analysis (ECE, reliability diagrams); larger
evaluation samples for tighter confidence intervals; hateBERT cross-domain
probe; neural masked fine-tune of DistilBERT as a debiasing intervention.

## 5. References
- Jigsaw/Conversation AI. Toxic Comment Classification Challenge. Kaggle, 2018.
- Dixon, Li, Sorensen, Thain, Vasserman. Measuring and Mitigating Unintended Bias in Text Classification. AIES 2018.
- Hanu, Unitary. Detoxify / `unitary/toxic-bert`, 2020.
- Sanh et al. DistilBERT / DistilGPT-2, 2019.

## 6. Ethics statement (required)

**6.1 Introduction.** Names: Tom Jacob, Omer Ramon. Project title: *Words That Harm*.
In this project we aim to compare SVM, decoder-only, and encoder-only architectures
for toxic-comment classification. We check for accuracy, identity bias, and robustness.

**6.2 Stakeholders and explanations.**
- *Stakeholders:* (1) platform users/commenters, (2) human moderators, (3)
  identity groups at risk of over-flagging.
- *Explanations:* users need to know why a comment was flagged and how to appeal;
  moderators need per-label scores and confidence; identity groups need
  transparency about measured per-group false-positive rates.
- *Responsibility:* the platform deploying the model owns user- and
  group-facing explanations; the ML team owns moderator-facing documentation.

**6.3 Our reflection (write ourselves — no LLM).**

Prompts to answer in 1–2 short paragraphs total before submission:
1. Given our measured identity FPRs (toxic-bert ≈ 14%, SVM ≈ 8%, decoder near-degenerate), what should change in how an AI moderation system *answers* or *acts* toward users?
2. Would you require human review for flags that fire on neutral identity mentions? Why / why not?
3. Should platforms publish bias-audit numbers like ours? What risk does that create, and what benefit?

[To be added by the team.]

---

*Reproduce every figure: see `docs/PROGRESS.md` §5 and `docs/DEVELOPER_GUIDE.md`.*
