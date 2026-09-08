# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

**Authors:** Omer Ramon, Tom Jacob · Technion EE046211 Deep Learning
**Repo:** https://github.com/Tom-Jacob/DeepProjectHate

> Draft for the ≤8-page report. Figures are in `report/figures/`.
>
> **Status.** §3.1 (accuracy), §3.2 (fairness), §3.3 (robustness, now
> **bidirectional** — evasion AND FP-induction), §3.4 (masking), and §3.10
> (same-checkpoint classifier vs. prompting, new) are complete with real
> numbers for every model with local weights (E2, E0, E1, E1-p, D1, D2) at
> n=800 (comparison), exhaustive (fairness probe), and n=100 (robustness,
> masking) — **not** the originally planned 5k/1k/1k Colab scale, a
> deliberate reduction to fit the time budget. §3.6 (distillation), §3.7
> (decoder fine-tune), §3.8 (fairness discussion) are complete with
> preliminary numbers **reported by Tom, not independently verified from
> this repo** (no weight file, result JSON, or executed notebook backs them
> in this checkout — see `STATUS.md`). The `distilbert-student` (S1) /
> `distilgpt2-d1-ft` (D1-ft) rows stay "—" throughout for the same reason.
> **Still pending:** §3.5 (agreement, needs D1-ft), §3.9 (FPR-regularized
> fine-tune, needs distilbert-student), perturbation-augmented fairness,
> full 63k-test confirmation. Ethics §6.3 to write (human-only).

---

## Abstract

We audit toxic-comment detectors on the Jigsaw 2018 dataset across model
configurations spanning four architectural families evaluated on the **same
data**: a specialized BERT encoder (`unitary/toxic-bert`), a non-neural
TF-IDF + linear-SVM baseline we train ourselves, a DistilGPT-2 decoder
(zero-shot and fine-tuned), a DistilBERT student trained via knowledge
distillation from the encoder, and — the controlled centerpiece for the
classifier-vs-prompting question — the **same** `bert-base-uncased` checkpoint
used two ways: fine-tuned with a classification head, and prompted zero-shot
via its native masked-LM head. This last pair isolates *method* from
*architecture family*, which the toxic-bert-vs-DistilGPT-2 comparison alone
cannot. Rather than chasing accuracy, we stress-test each model for
**identity bias**, **robustness to obfuscation**, and the **mechanism** behind
its errors. We find that (i) toxicity detection is substantially lexical — a
linear model reaches ~93% of the encoder's F1 at ~3300× lower latency, though
the encoder keeps a real edge, not just a latency trade-off; (ii) the encoder's
more decisive advantage is robustness to obfuscation, not accuracy — and that
robustness is not free with the architecture, since our own generic
`bert-base` classifier (same architecture, same size as toxic-bert) is nearly
as brittle as the linear model — and, measured bidirectionally, toxic-bert's
robustness itself is direction-dependent: the same surface noise that barely
evades a real toxic comment (evasion 5-15%) gets a neutral comment wrongly
flagged as toxic 65-91% of the time; (iii) zero-shot decoder prompting at
DistilGPT-2 scale is near-chance, but light fine-tuning recovers usable
discrimination (F1 0.59, ROC-AUC 0.97); on the identical `bert-base-uncased`
checkpoint, a trained classifier head dramatically beats zero-shot masked-LM
prompting too (F1 0.68 vs. 0.00 at the default threshold), though BERT's
zero-shot prompt still ranks far better (ROC-AUC 0.72) than GPT-2's zero-shot
prompt (ROC-AUC ≈ 0.4); (iv) soft knowledge distillation matches the teacher's
F1 while *reducing* identity bias; and (v) identity bias is present in every
classifier and is data-driven, but its *mechanism* is not uniform — masking
the identity token erases the bias for toxic-bert and the SVM (a clean lexical
shortcut) but *increases* it for our generic `bert-base` classifier.

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
(H2) smaller/compressed models lose accuracy *and* amplify bias; (H3) decoder /
zero-shot prompting is weaker and more format-sensitive than a discriminative
head — tested across architecture families (DistilGPT-2 vs. toxic-bert) and,
more tightly, on the identical `bert-base-uncased` checkpoint used both ways.

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

We compare eight model configurations spanning four architectural families.
Every model exposes the same `predict_toxic_proba` interface so all
evaluations are architecture-agnostic.

| ID | Model | Family | Params | Source | Classification mechanism |
|----|-------|--------|--------|--------|--------------------------|
| E2 | toxic-bert | BERT encoder | 109M | `unitary/toxic-bert` (HF Hub) | 6 sigmoid heads; we read the `toxic` logit |
| E0 | TF-IDF + LinearSVM | Non-neural | ~50k | Trained by us | TF-IDF (1-2 grams, 50k features) + calibrated LinearSVC |
| E1 | bert-base-classifier | BERT encoder | 109M | Fine-tuned by us | `bert-base-uncased` + binary classification head, class-weighted loss (12k-row subsample, CPU) |
| E1-p | bert-base-mlm-prompt | BERT encoder (zero-shot) | 109M | `bert-base-uncased` (HF Hub), untrained | Prompted: "Comment: {text}\nThis comment is [MASK]." Fill-mask logit for `toxic` vs `fine` |
| D1 | DistilGPT-2 yes/no (zero-shot) | GPT-2 decoder | 82M | `distilgpt2` (HF Hub) | Prompted: "Is this comment toxic? Answer yes or no." Log-prob of ` yes` vs ` no` |
| D2 | DistilGPT-2 fill-in-blank (zero-shot) | GPT-2 decoder | 82M | `distilgpt2` (HF Hub) | Prompted: "This comment is ___." Log-prob of ` toxic` vs ` not toxic` |
| D1-ft | DistilGPT-2 yes/no (fine-tuned) | GPT-2 decoder | 82M | Fine-tuned from D1 | Same D1 prompt; causal LM loss on answer tokens only (20k train, 4 epochs) |
| S1 | DistilBERT student | DistilBERT | 67M | Distilled from toxic-bert | Soft KD (T=2, $\alpha$=0.7); 6 sigmoid heads |

**E1 and E1-p are the SAME pretrained checkpoint** (`bert-base-uncased`) used two
ways — a trained classification head vs. zero-shot masked-LM prompting — giving
a controlled test of "dedicated classifier vs. LM prompting" (H3) that isn't
confounded by architecture family the way E2-vs-D1 is (BERT vs. GPT-2). E1 also
restores a specialization control for H1: same architecture as toxic-bert (E2),
generic instead of in-domain-specialized.

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

**Decoder input truncation.** DistilGPT-2 has a 1024-token context window
(learned positional embeddings). When a long Jigsaw comment is wrapped in the
D1 or D2 prompt template the total token count can exceed this limit. Tokens
beyond position 1024 receive out-of-range positional indices, producing
corrupted logits and unreliable scores. To prevent this, we truncate the
tokenized prompt from the *left* so that `prompt + continuation` never exceeds
1024 tokens. Left-truncation preserves the tail of the prompt — the `"Answer:"`
or `"This comment is"` suffix — which is the part the causal model conditions
on when scoring the next token. The beginning of very long comments (roughly
the top 1% of Jigsaw by length) is clipped, but the question/continuation
tokens the model needs to make its prediction remain intact. Encoder models
(BERT, DistilBERT) use a separate `truncation=True` parameter during
tokenization and are unaffected.

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

Following Dixon et al. (2018), we construct neutral sentences by filling
template carriers with 23 identity terms across 6 categories (religion, race,
orientation, gender, disability, plus two **neutral baselines**: "human" and
"normal"). To ensure the probe covers a realistic range of comment lengths we
use three tiers of templates: **10 short** one-liners ("I am {g}.", "My friend
is {g}.", ~5 words), **3 medium** forum-comment-length paragraphs (~50 words),
and **3 long** Wikipedia-talk-page-style discussions (~140 words). The medium
and long templates are styled after real Jigsaw-source comments — collaborative
editing discussions that mention an identity term in a clearly non-toxic
context. This gives 16 templates × 23 groups = **368 probe sentences** total.
By construction none is toxic, so any positive prediction is a false positive.
We compare per-group FPR against 8 length-matched no-identity control sentences
to verify that the bias is identity-driven, not template- or length-driven.

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

**Protocol.** Every model with local weights (E2, E0, E1, E1-p, D1, D2) is
scored on the **same fixed 800-comment val/test subset** (seeded, CPU) so every
model sees identical data. We measure F1, PR-AUC, ROC-AUC, and inference
latency (ms/comment, CPU). S1 (`distilbert-student`) and D1-ft
(`distilgpt2-d1-ft`) need Tom's Colab-trained weights, not present in this
checkout — their rows stay "—" until run alongside the still-pending full
5,000/63,978-comment Colab pass (`notebooks/15_full_eval_colab.ipynb`), which
also confirms these CPU/n=800 numbers at full scale and on GPU latency.

**Results (fair comparison, n=800, CPU).**

| model | params | F1 | PR-AUC | ROC-AUC | ms/comment |
|---|--:|:--:|:--:|:--:|--:|
| toxic-bert (E2) | 109M | 0.752 | 0.835 | 0.979 | 175 |
| bert-base-classifier (E1) | 109M | 0.675 | 0.791 | 0.961 | 278 |
| tfidf-svm (E0) | 50k | 0.703 | 0.784 | 0.958 | 0.1 |
| bert-base-mlm-prompt (E1-p) | 109M | 0.000 | 0.227 | 0.722 | 283 |
| distilgpt2-d2 (zero-shot) | 82M | 0.202 | 0.098 | 0.457 | 645 |
| distilgpt2-d1 (zero-shot) | 82M | 0.163 | 0.100 | 0.389 | 649 |
| distilbert-student (KD) | 67M | — | — | — | — |
| distilgpt2-d1-ft (fine-tuned) | 82M | — | — | — | — |

**Interpretation.** The linear model (E0) is close to toxic-bert on F1 but
toxic-bert keeps a real edge on every metric at this scale — most of the task
is still substantially lexical, but "the SVM matches/beats the encoder" does
not fully hold up once the sample is large enough. The zero-shot decoders are
near chance, consistent with H3: a small language model cannot discriminate
toxicity without task-specific supervision. E1-p (zero-shot BERT prompting)
looks similarly weak by F1 but has a far better ROC-AUC than the GPT-2
decoders — see §3.10. The D1 fine-tune (§3.7, separate protocol) recovers
usable discrimination but still trails the encoder and SVM.

### 3.2 Fairness: per-identity FPR
*Figure: `fairness_fpr_heatmap.png`.*

**Goal.** Test H1 (specialized encoder trades fairness for accuracy) by
measuring how often each model wrongly flags neutral identity-mention sentences.

**Protocol.** Score the 368-sentence identity probe (§2.5, including "human"
and "normal" neutral baselines) plus 8 no-identity controls with each model
(with local weights) at threshold 0.5. Report overall mean FPR across all
groups, mean FPR restricted to the two neutral-baseline groups, and control
FPR. S1 and D1-ft again need Tom's Colab weights and stay pending; so does the
perturbation-augmented version of this probe (`--perturb`).

**Results (unperturbed, exhaustive probe).**

| model | overall identity FPR | neutral FPR | control FPR |
|---|:--:|:--:|:--:|
| tfidf-svm (E0) | 0.049 | 0.000 | 0.000 |
| toxic-bert (E2) | 0.087 | 0.000 | 0.000 |
| bert-base-classifier (E1) | 0.103 | 0.000 | 0.000 |
| bert-base-mlm-prompt (E1-p) | 0.000 | 0.000 | 0.000 |
| distilgpt2-d2 | 1.000 | 1.000 | 1.000 |
| distilgpt2-d1 | 0.899 | 0.813 | 1.000 |
| distilbert-student | — | — | — |
| distilgpt2-d1-ft | — | — | — |

**Perturbed fairness.** *(Still pending — not run in this pass; needs
`scripts/run_fairness.py --perturb`, feasible on CPU but not run yet.)*

**Interpretation.** Control FPR is 0 for all trained classifiers, meaning false
positives are *specifically* identity-driven ("I am gay", "I am Muslim" are
flagged despite being non-toxic). The specialized encoder is **less fair** than
the linear baseline (H1 supported) — and, per §3.10, our own *generic* E1 is
less fair still, despite sharing toxic-bert's exact architecture, weakening a
naive reading of H1. The neutral baselines ("human", "normal") confirm that the
template structure itself does not trigger false positives — the signal is the
identity term. Zero-shot decoders (D1 and D2) flag essentially everything.

### 3.3 Robustness to obfuscation (bidirectional)
*Figures: `robustness_evasion_heatmap.png`, `robustness_fp_induction_heatmap.png`.*

**Goal.** Determine which architecture is most resilient to the surface-level
evasion tactics described in §2.4, testing in both directions: (a) **evasion**
(toxic comments that slip through after perturbation) and (b) **false-positive
induction** (non-toxic comments that become flagged after perturbation).

**Protocol.** (a) *Toxic probe:* take 100 genuinely-toxic test comments that
each model flags as toxic (score $\ge$ 0.5), apply each perturbation and
re-score. Report evasion rate. (b) *Non-toxic probe:* take the correctly-
classified non-toxic test comments, apply each perturbation, and report the
fraction that flip to toxic (FP induction rate). Both directions are now run
(at n=100, not the originally-planned n=1,000 — a deliberate reduction to fit
the time budget; direction, not just magnitude, is the finding here).

$$\text{FP induction} = \frac{|\{x : P_\text{orig}(x) < 0.5 \;\wedge\; P_\text{pert}(x) \ge 0.5\}|}{|\{x : P_\text{orig}(x) < 0.5\}|}$$

**Results (evasion — toxic probe, n=100).**

| model | baseline det. | leetspeak | space-out | punct-noise |
|---|:--:|:--:|:--:|:--:|
| toxic-bert (E2) | 0.930 | 0.280 | 0.054 | 0.151 |
| bert-base-classifier (E1) | 0.890 | 0.719 | 0.191 | 0.225 |
| tfidf-svm (E0) | 0.760 | 0.711 | 0.750 | 0.750 |
| distilgpt2-d1 | 0.580 | 0.103 | 0.172 | 0.172 |
| bert-base-mlm-prompt (E1-p) | 0.000 | n/a | n/a | n/a |
| distilbert-student | — | — | — | — |

**Results (FP induction — non-toxic probe, n=100).**

| model | baseline FP | leetspeak | space-out | punct-noise |
|---|:--:|:--:|:--:|:--:|
| toxic-bert (E2) | 0.110 | 0.494 | **0.910** | 0.652 |
| bert-base-classifier (E1) | 0.090 | 0.000 | 0.703 | 0.670 |
| tfidf-svm (E0) | 0.050 | 0.000 | 0.021 | 0.021 |
| distilgpt2-d1 | 0.840 | 0.438 | 0.375 | 0.313 |
| bert-base-mlm-prompt (E1-p) | 0.010 | 0.000 | 0.000 | 0.000 |

**Interpretation.** toxic-bert is far the most robust to leetspeak *evasion*
— its WordPiece tokenizer and contextual embeddings survive most surface
noise. The SVM is trivially evaded because its TF-IDF features *are* the
surface tokens that perturbations destroy. Notably (§3.10), our own E1 —
identical architecture and size to toxic-bert — is nearly as brittle to
leetspeak as the SVM, showing the robustness gap tracks toxic-bert's specific
fine-tuning rather than "being a BERT encoder" in general.

**The FP-induction direction flips the story.** toxic-bert's evasion
robustness does **not** carry over to resisting false-positive induction: a
neutral comment that's merely space-out or punct-noise perturbed gets flagged
as toxic 91% / 65% of the time (vs. only 11% baseline) — far worse than the
SVM (2% / 2%), which barely reacts to any of these perturbations in either
direction. E1 shows a similar over-reaction (70% / 67%). So "toxic-bert is
robust" is direction-dependent: hard to trick into missing a real toxic
comment, but easy to trick into over-flagging an innocent one with the exact
same surface noise — the opposite failure mode from what evasion alone would
suggest, and a genuinely new fairness/reliability concern distinct from the
identity-specific bias in §3.2. The SVM's stability in this direction is
consistent with its overall pattern (§3.2, §3.4): it is a **weak but
consistent** model — never very sensitive to anything unusual, whether that
helps (FP induction) or hurts (evasion) it. `bert-base-mlm-prompt`'s near-zero
FP induction is again the same calibration floor effect as elsewhere
(it essentially never says "toxic" regardless of input).

### 3.4 Masking intervention
*Figures: `masking_inference_ablation.png`, `masking_svm_tradeoff.png`.*

**Goal.** Identify the *mechanism* behind identity-term false positives (§3.2):
is the model relying on the identity token itself, or on surrounding context?

**Protocol.** (a) *Inference masking*: replace the identity term in each probe
sentence with `[MASK]` (encoder) or empty string (SVM) and re-measure FPR,
now for every model with local weights (E2, E0, E1, E1-p). (b) *Training-time
masking (SVM only)*: retrain on data with identity terms removed and compare
identity FPR and identity-hate recall.

**Results.**

| model | identity FPR, with term | identity FPR, masked |
|---|:--:|:--:|
| toxic-bert (E2) | 0.087 | 0.000 |
| tfidf-svm (E0) | 0.049 | 0.000 |
| bert-base-classifier (E1) | 0.103 | 0.188 |
| bert-base-mlm-prompt (E1-p) | 0.000 | 0.000 |

toxic-bert and the SVM — a deep transformer and a bag-of-words — both rely on
the **identity token itself**: masking it collapses their FPR to zero, a clean
lexical shortcut. **E1 breaks this pattern** (see §3.10): masking *increases*
its identity FPR instead, so the "mask the identity term and bias disappears"
story is not universal. Training-time masking of the SVM does *not* reduce FPR
(0.049 $\to$ 0.063) but costs identity-hate recall
(0.781 $\to$ 0.676).

**Interpretation.** For toxic-bert and the SVM the bias is a single-token
shortcut learned from the training data, not an architectural artifact. Naive
token removal during training is an insufficient and harmful "fix" — it
sacrifices real identity-hate detection for negligible fairness improvement.
E1's opposite reaction to masking (§3.10) shows this single-token-shortcut
story does not generalize to every classifier trained on this data.

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

### 3.9 FPR-regularized fine-tuning
*Colab: `notebooks/16_fpr_finetuning.ipynb`. Figure: `fpr_finetune_tradeoff.png`.*

**Goal.** Reduce identity-probe FPR without destroying accuracy, using an
explicit loss penalty rather than naive data masking.

**Motivation.** The masking experiment (§3.4) showed that identity bias is a
lexical shortcut: removing the identity token at inference zeroes FPR, but
removing it at training time *hurts* recall (0.78 → 0.68) with negligible
fairness gain. We need a smarter approach that targets FPR directly in the
training objective.

**Method.** Start from the trained distilbert-student (§3.6) and run a
**second-stage fine-tune** with a combined loss:

$$\mathcal{L} = \text{BCE}(\text{task batch}) + \gamma \cdot \frac{1}{|P|}\sum_{x \in P} \sigma\bigl(f(x)_\text{toxic}\bigr)$$

where $P$ is the 368-sentence identity probe set (§2.5) and $\sigma$ is the
sigmoid. The BCE term preserves accuracy on the main Jigsaw task; the FPR
penalty pushes the model to score identity-probe sentences as non-toxic. We
sweep $\gamma \in \{0.1, 0.5, 1.0, 2.0\}$ with 2 epochs, 20k train examples,
and lr = 1e-5.

**Results.** *(Placeholder — update after Colab run.)*

| variant | Test F1 (toxic) | Identity FPR | IH recall |
|---|:--:|:--:|:--:|
| baseline (γ=0) | — | — | — |
| γ=0.1 | — | — | — |
| γ=0.5 | — | — | — |
| γ=1.0 | — | — | — |
| γ=2.0 | — | — | — |

**Interpretation.** *(Pending results.)* The hypothesis is that a moderate γ
can push FPR toward zero while keeping F1 within ±0.01 of baseline — a much
better tradeoff than naive masking, which sacrifices recall for negligible
fairness improvement. If confirmed, FPR-regularized fine-tuning offers a
practical debiasing recipe: train normally, then run a short second stage with
the probe set in the loss.

### 3.10 Same checkpoint, two methods: classifier vs. prompting (E1 vs. E1-p)
*Rows for `bert-base-classifier` (E1) / `bert-base-mlm-prompt` (E1-p) in the
§3.1–3.4 tables; pairwise agreement via `scripts/run_bertbase_agreement.py`.*

**Goal.** §3.1's toxic-bert-vs-DistilGPT-2 comparison, and even the D1-ft
comparison in §3.7, confound two things: architecture family (encoder vs.
decoder) and method (trained head vs. prompting). E1 and E1-p hold
architecture **and pretrained weights** fixed — literally the same
`bert-base-uncased` checkpoint — and vary only the method, giving the most
controlled test of Roy's original question, "dedicated classifier vs. LM
prompting" (H3).

**Protocol.** E1: fine-tune `bert-base-uncased` with a binary classification
head on a stratified 12k-row Jigsaw subsample (class-weighted loss for the
~10% positive rate; CPU-only, ~2h). E1-p: the same, untrained checkpoint,
prompted via its native masked-LM head with `"Comment: {text}\nThis comment is
[MASK]."`, scoring the fill-mask logit for `toxic` vs. `fine`. Both evaluated
with the same n=800 comparison set, exhaustive fairness probe, and n=100
robustness probe as §3.1–3.3.

**Results.** From §3.1 (test, n=800): E1 F1=0.675, PR-AUC=0.791, ROC-AUC=0.961
vs. E1-p F1=0.000, PR-AUC=0.227, ROC-AUC=0.722 — on the exact same weights, the
trained head dominates the zero-shot prompt on every metric. E1-p's F1=0 is a
calibration artifact, not a failure to discriminate: it almost never crosses
the 0.5 threshold, yet its ROC-AUC is far above DistilGPT-2's zero-shot D1/D2
(≈0.4), showing BERT's bidirectional masked-LM objective gives a much better
zero-shot signal than GPT-2's causal one — naive thresholding just hides it.

Pairwise agreement (n=800, `scripts/run_bertbase_agreement.py`): the two
methods agree on **82.8%** of comments. Accuracy against gold looks
deceptively close (E1 0.908, E1-p 0.885) — but E1-p's accuracy is mostly the
~88–90% non-toxic base rate. Of the 138 disagreements, E1 is consistently the
one catching real toxicity E1-p misses, e.g. "Hey fuck you wikipedia! I do
exist!" (gold=1, E1=1, E1-p=0).

**Specialization (H1), independent of toxic-bert.** From §3.2: generic E1
(identity FPR 0.103) is *less* fair than specialized toxic-bert (0.087) —
the same direction as the E0 (SVM) comparison, but here
with architecture family held fixed too (both are BERT-base-sized encoders).
This weakens a naive reading of H1: *less* specialization was less fair here,
suggesting toxic-bert's larger-scale, higher-quality fine-tuning — not
specialization per se — is what limits its bias and gives it the robustness
edge in §3.3, where E1 is nearly as brittle to leetspeak as the SVM despite
sharing toxic-bert's exact architecture.

**Mechanism divergence (§3.4).** Masking the identity token collapses identity
FPR to zero for toxic-bert and the SVM (a clean lexical shortcut) but
*increases* E1's FPR instead — inserting a literal `[MASK]` token (which
essentially never appears in E1's natural-comment training data) seems to push
an out-of-distribution input toward "toxic" rather than neutral. Identity bias
is data-driven across every model here, but *how* it is encoded is
model-specific, not a single universal shortcut.

## 4. Conclusions and future work

1. **Toxicity detection is substantially lexical, but not purely so** — a
   50k-feature linear model reaches ~93% of a 109M encoder's F1 at ~3300× lower
   latency, yet the encoder retains a real, consistent edge; "bigger" buys a
   bounded, not a transformative, improvement.
2. **The encoder's real value is robustness to obfuscation, not raw accuracy —
   and that robustness is not free with the architecture, nor is it
   direction-independent.** Our own generic `bert-base-classifier` (E1) —
   same architecture and parameter count as toxic-bert — is nearly as
   brittle to leetspeak as the linear SVM; toxic-bert's edge tracks its
   specific (larger-scale, in-domain) fine-tuning, not "being a BERT".
   Moreover, toxic-bert's evasion robustness does **not** carry over to the
   opposite direction: a neutral comment perturbed with space-out or
   punct-noise gets wrongly flagged as toxic 91% / 65% of the time (vs. 11%
   baseline) — far worse than the SVM (2% either way), which is simply a
   *weak but consistent* model, insensitive to surface noise in both
   directions rather than "robust" in a meaningful sense.
3. **Zero-shot decoder prompting fails at DistilGPT-2 scale** — near chance and
   often degenerate; a **D1 fine-tune** (§3.7) recovers usable discrimination
   (test F1 0.59, ROC-AUC 0.97) without changing architecture, though identity
   FPR stays high (0.51). **The same conclusion holds even more cleanly on a
   single matched checkpoint (§3.10):** `bert-base-classifier` dominates
   `bert-base-mlm-prompt` on every metric using *identical* pretrained weights,
   isolating method from architecture — though BERT's zero-shot prompt still
   ranks meaningfully better (ROC-AUC 0.72) than GPT-2's zero-shot prompt
   (≈0.4), so architecture and method both matter independently.
4. **Identity bias is data-driven but not uniform in mechanism.** It appears
   even in the linear model, and masking the identity token erases it there
   and for toxic-bert (a lexical shortcut) — but *increases* it for our
   generic `bert-base-classifier` (§3.10), whose over-flagging is evidently
   not reducible to a single token. Fixing bias via naive training-masking
   (E0) trades recall for little gain.
5. **Distillation (soft KD):** the DistilBERT student matches toxic-bert toxic F1
   (0.772 vs 0.770) at lower latency and **lower** identity FPR (0.062 vs 0.152)
   on our probe — compression need not amplify bias (**H2** partially rejected).
6. **Specialization (H1) is not the whole story.** Generic `bert-base-classifier`
   is *less* fair than specialized toxic-bert (§3.2, §3.10) — consistent with
   the SVM comparison, but here with architecture family held fixed too. This
   points at toxic-bert's training scale/quality, not specialization per se,
   as the driver of both its fairness and robustness edge.

7. **FPR-regularized fine-tuning (§3.9):** a second-stage loss penalty on the
   identity probe directly targets the bias mechanism identified by the masking
   experiment, offering a more principled fix than naive token removal.

**Future work.** Calibration analysis (ECE, reliability diagrams); hateBERT
cross-domain probe; few-shot decoder prompting; extending FPR regularization
to the encoder itself (including E1); fairness-constrained optimization with
per-group FPR targets; independently reproducing the distillation and decoder
fine-tune results (§3.6–3.7) from saved artifacts rather than reported
numbers; the still-pending full evaluation pass (63k-test confirmation at the
Colab scale, D1-ft agreement, perturbation-augmented fairness).

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
