# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

**Authors:** Omer Ramon, Tom Jacob · Technion EE046211 Deep Learning
**Repo:** https://github.com/Tom-Jacob/DeepProjectHate

> Draft for the ≤8-page report. Figures are in `report/figures/`. Numbers below are
> from the full-scale reruns (test/val n=800 for the comparison table, n=300 toxic
> comments for robustness, n=400/800 for the agreement analyses; the fairness probe
> is exhaustive over 21 groups × 10 templates, not sampled).
> The ethics section §6.3 must be written by us (no LLM) — see note there.

---

## Abstract

We audit toxic-comment detectors on the Jigsaw 2018 dataset across five model
configurations evaluated on the **same data**: a specialized BERT encoder
(`unitary/toxic-bert`), a non-neural TF-IDF + linear-SVM baseline we train
ourselves, a DistilGPT-2 decoder used as a zero-shot classifier via prompting,
and — the controlled centerpiece — the **same** `bert-base-uncased` checkpoint
used two ways: fine-tuned with a classification head, and prompted zero-shot via
its native masked-LM head. This last pair isolates *method* (trained classifier
vs. LM prompting) from *architecture family*, which the toxic-bert-vs-DistilGPT-2
comparison alone cannot. Rather than chasing accuracy, we stress-test each model
for **identity bias**, **robustness to obfuscation**, and the **mechanism**
behind its errors. We find that (i) toxicity detection is substantially lexical —
a 50k-feature linear model reaches ~93% of toxic-bert's F1 and ~94% of its PR-AUC
at ~3300× lower latency, though at full scale the transformer keeps a real,
non-trivial edge on every accuracy metric rather than being matched outright;
(ii) the specialized encoder's more decisive advantage is robustness to
obfuscation, not raw accuracy; (iii) on the identical `bert-base-uncased`
checkpoint, a trained classifier head dramatically beats zero-shot masked-LM
prompting (F1 0.68 vs. 0.00 at the default threshold), and generative/zero-shot
prompting more generally (DistilGPT-2 included) is near-chance and degenerate at
this scale, though BERT's bidirectional zero-shot prompt still ranks far better
(ROC-AUC 0.72) than GPT-2's causal one (ROC-AUC ≈ 0.4); and (iv) identity bias is
present in every classifier — including our own generic encoder, which is
*less* fair than the specialized one — but the underlying mechanism is not
uniform: masking the identity token erases the bias for toxic-bert and the SVM
(a clean lexical shortcut) but *increases* it for our generic classifier,
showing bias is data-driven yet model-specific in how it manifests.

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
(H2) smaller/compressed models lose accuracy *and* amplify bias; (H3) prompting
an LM zero-shot is weaker and more format-sensitive than a trained discriminative
head — tested twice: across architecture families (DistilGPT-2 vs. toxic-bert)
and, more tightly, on the identical `bert-base-uncased` checkpoint used both ways.

## 2. Method

**Task & data.** Jigsaw 2018, binary target `toxic` (multi-label labels retained
for the encoder↔decoder analysis). Split 85/15 train/val from `train.csv`
(stratified), test = labeled `test.csv` rows (135,635 / 23,936 / 63,978).

**Architectures (same data, fair comparison).**
- **Encoder (specialized)** — `unitary/toxic-bert` (BERT, 109M), fine-tuned on Jigsaw; 6 sigmoid heads, we read the `toxic` logit.
- **Encoder (generic, trained by us)** — `bert-base-uncased` (BERT, 109M) fine-tuned with a binary classification head on a stratified Jigsaw subsample (class-weighted loss for the ~10% positive rate; CPU-only, so trained on a subsample rather than the full 135k-row train split).
- **Encoder (generic, zero-shot prompt)** — the *same* `bert-base-uncased` checkpoint, untrained, prompted via its native masked-LM head: `"Comment: {text}\nThis comment is [MASK]."`, scored by the fill-mask logit for "toxic" vs. a non-toxic token. Same checkpoint as the line above; only the method differs.
- **Non-neural** — TF-IDF (1–2 grams, 50k) + calibrated LinearSVM, trained by us on the same split.
- **Decoder** — DistilGPT-2 (82M) as a zero-shot classifier: D1 = "Is this toxic? yes/no", D2 = fill-in-blank "This comment is ___", scored by continuation log-probability.

**Unified interface.** Every model is reduced to a single `P(toxic)` so all
evaluations are architecture-agnostic (`predict_toxic_proba`). We report
**parameter count and inference latency** alongside accuracy for a fair
capacity comparison.

**Metrics & probes.**
- Accuracy: F1, PR-AUC, ROC-AUC at threshold 0.5.
- **Fairness:** per-identity false-positive rate on neutral identity-mention
  templates (Dixon-style), across 21 groups; a no-identity control set.
- **Robustness:** evasion rate — fraction of genuinely-toxic comments that slip
  through after surface perturbations (typos, leetspeak, spacing, repetition,
  casing, punctuation).
- **Masking intervention:** (a) at inference, replace the identity term with
  `[MASK]` and re-measure FPR; (b) retrain the SVM with identity terms removed
  and measure the fairness/recall trade-off.
- **Agreement:** encoder vs decoder prediction agreement and each vs gold.

## 3. Experiments and results

### 3.1 Accuracy vs. efficiency (fair comparison)
*Figures: `model_comparison_bars_test.png`, `accuracy_latency_test.png`. Test split, n=800.*

| model | params | F1 | PR-AUC | ROC-AUC | ms/comment |
|---|--:|:--:|:--:|:--:|--:|
| toxic-bert (BERT, specialized) | 109M | **0.752** | **0.835** | **0.979** | 662 |
| bert-base-classifier (BERT, generic, ours) | 109M | 0.675 | 0.791 | 0.961 | 802 |
| tfidf-svm (non-neural) | 50k | 0.703 | 0.784 | 0.958 | **0.2** |
| bert-base-mlm-prompt (BERT, zero-shot) | 109M | 0.000 | 0.227 | 0.722 | 800 |
| distilgpt2-d2 (decoder, zero-shot) | 82M | 0.202 | 0.098 | 0.457 | 520 |
| distilgpt2-d1 (decoder, zero-shot) | 82M | 0.163 | 0.100 | 0.389 | 841 |

At full scale, toxic-bert keeps a real edge over the linear SVM on every metric —
the earlier small-sample result that the SVM "beats" it on PR-AUC does not hold up.
But the gap is modest (the SVM reaches 93% of toxic-bert's F1, 94% of its PR-AUC)
at ~3300× lower latency, so most of the task is still substantially lexical; the
extra capacity buys a real but bounded improvement, not a different regime.

`bert-base-mlm-prompt` is the standout anomaly: **F1 = 0.000** — at the default
0.5 threshold it never predicts "toxic" — yet its **ROC-AUC (0.722) is far above**
DistilGPT-2's zero-shot prompting (0.39–0.46). The model *can* rank toxic
comments correctly; its output scale is just miscalibrated relative to a 0.5 cut.
This shows BERT's bidirectional masked-LM objective gives a much better zero-shot
signal than GPT-2's causal one, but naive thresholding hides it — accuracy alone
is a misleading metric here, exactly why we report PR-AUC/ROC-AUC alongside F1.
Both decoder variants remain near chance.

### 3.2 Fairness: per-identity FPR
*Figure: `fairness_fpr_heatmap.png`. Exhaustive over 21 identity groups × 10 templates.*

| model | overall identity FPR | control FPR |
|---|:--:|:--:|
| bert-base-mlm-prompt | 0.000* | 0.000 |
| tfidf-svm | 0.086 | 0.000 |
| toxic-bert | 0.152 | 0.000 |
| bert-base-classifier | 0.181 | 0.000 |
| distilgpt2-d1 | 0.867 | **1.000** |

\* Not genuine fairness — consistent with §3.1, `bert-base-mlm-prompt` almost
never crosses the 0.5 threshold at all, so it trivially never "fires" on the
probe either.

Control FPR is 0 for every classifier → false positives are *specifically*
identity-driven ("I am gay", "I am Muslim", "I am transgender" are flagged). The
specialized encoder is **less fair** than the linear baseline, replicating H1's
core claim — but our own **generic** `bert-base-classifier` is less fair still
(0.181, the worst non-degenerate model), on the *same* architecture as toxic-bert
and *less* specialized. Specialization alone does not explain the ranking here;
see §3.6. The decoder flags essentially everything (control FPR 1.0) — degenerate
as a classifier.

### 3.3 Robustness to obfuscation
*Figure: `robustness_evasion_heatmap.png`. 300 genuinely-toxic test comments.*
Evasion = flagged-toxic comment that slips through after perturbation (higher = more brittle).

| model | baseline det. | leetspeak | space-out | punct-noise |
|---|:--:|:--:|:--:|:--:|
| toxic-bert | 0.910 | **0.289** | 0.022 | 0.150 |
| bert-base-classifier | 0.857 | 0.735 | 0.175 | 0.222 |
| tfidf-svm | 0.700 | 0.752 | 0.748 | 0.748 |
| distilgpt2-d1 | 0.670 | 0.179 | 0.144 | 0.204 |
| bert-base-mlm-prompt | 0.003* | 1.000* | 1.000* | 1.000* |

\* Degenerate, not informative: `bert-base-mlm-prompt` detects almost nothing at
baseline (1/300), so "evasion" is undefined in spirit — there is nothing to evade.

toxic-bert is by far the most robust to leetspeak; the SVM is trivially evaded
(its features *are* surface tokens). Notably, **`bert-base-classifier` is nearly
as brittle as the SVM to leetspeak (0.735 evasion)** despite sharing toxic-bert's
exact architecture and parameter count — the robustness gap in §3.1–3.3 tracks
toxic-bert's specific (larger-scale, in-domain) fine-tuning, not "being a BERT
encoder" in general. The decoder's low evasion is an artifact of near-random,
insensitive predictions (baseline detection only 0.67), not genuine robustness.
No model is meaningfully fooled by ALL-CAPS alone (omitted above; ~0 evasion
everywhere).

### 3.4 Masking intervention (why they over-flag identities)
*Figures: `masking_inference_ablation.png`, `masking_svm_tradeoff.png`.*

Inference-time masking (replace the identity term with `[MASK]`):

| model | identity FPR, with term | identity FPR, masked |
|---|:--:|:--:|
| toxic-bert | 0.152 | **0.000** |
| tfidf-svm | 0.086 | **0.000** |
| bert-base-classifier | 0.181 | **0.300** (↑) |
| bert-base-mlm-prompt | 0.000 | 0.000 (mean P(toxic) still drops 0.092→0.048) |

toxic-bert and the SVM — a deep transformer and a bag-of-words — both rely on
the **identity token itself**: masking it collapses their false-positive rate to
zero, a clean lexical shortcut. **`bert-base-classifier` breaks this pattern: masking
*increases* its identity FPR** (0.181 → 0.300), and its mean P(toxic) rises too
(0.225 → 0.249), not just crosses the threshold by noise. So the "mask the
identity term and bias disappears" story is not universal — our generic
classifier's over-flagging is not simply keyed to the identity word; inserting a
literal `[MASK]` token (which essentially never appears in its natural-comment
training data) appears to push an out-of-distribution input toward "toxic" rather
than neutral, a different, less well-understood failure mode. Identity bias is
data-driven across every model here, but *how* it is encoded is model-specific.

Training-time masking of the SVM does *not* reduce FPR (0.086 → 0.100) but costs
identity-hate recall (0.781 → 0.676) at roughly stable general-toxicity recall
(0.659 → 0.646): naive removal is an insufficient and harmful "fix".

### 3.5 Encoder vs. decoder agreement
*Figures: `enc_dec_gold_agreement.png`, `enc_dec_agreement.png`. (test n=400, 6 labels.)*

Overall encoder↔decoder agreement is only **0.207**. Against gold on `toxic`, the
encoder scores **0.930** but the decoder only **0.245** — the two barely agree not
because they capture different *valid* signals, but because the decoder is
degenerate: it labels almost every comment toxic. Of 400 comments the two
disagree on `toxic` **292** times (73%), and the disagreements are overwhelmingly
decoder false positives on benign text.

| comment (truncated) | gold | encoder | decoder |
|---|:--:|:--:|:--:|
| ":Replied here:" | 0 | 0 | 1 |
| "US New Citizenship Test. In October of 2008 …" | 0 | 0 | 1 |
| "This is your last warning. You will be blocked …" | 0 | 0 | 1 |

Takeaway: cross-architecture agreement is a poor proxy for correctness here — a
strong discriminative encoder and a degenerate generative prompt "disagree" almost
everywhere, and the encoder is right nearly every time.

### 3.6 Same checkpoint, two methods: classifier vs. prompting
*Rows for `bert-base-classifier` / `bert-base-mlm-prompt` in the §3.1–3.4 tables;
pairwise agreement via `scripts/run_bertbase_agreement.py`.*

Section 3.1's toxic-bert-vs-DistilGPT-2 comparison confounds two things:
architecture family (encoder vs. decoder) and method (trained head vs. zero-shot
prompting). `bert-base-classifier` and `bert-base-mlm-prompt` hold architecture
fixed — literally the same pretrained weights — and vary only the method, giving
a controlled test of Roy's original question, "dedicated classifier vs. LM
prompting."

From §3.1 (test, n=800): `bert-base-classifier` F1=0.675, PR-AUC=0.791, ROC-AUC=0.961
vs. `bert-base-mlm-prompt` F1=0.000, PR-AUC=0.227, ROC-AUC=0.722 — on the exact
same weights, the trained head dominates the zero-shot prompt on every metric.
**H3 holds, and more cleanly than the toxic-bert-vs-DistilGPT-2 comparison**,
since architecture and pretraining are held fixed here.

Pairwise agreement (n=800, `scripts/run_bertbase_agreement.py`): the two methods
agree on **82.8%** of comments. Accuracy against gold looks deceptively close
(classifier 0.908, prompt 0.885) — but the prompt's accuracy is mostly the
~88–90% non-toxic base rate, since (per §3.1) it almost never predicts "toxic" at
all. Of the 138 disagreements, the classifier is consistently the one catching
real toxicity the prompt misses, e.g.:

| comment (truncated) | gold | classifier | prompt |
|---|:--:|:--:|:--:|
| "…Did you get raped your brai[n]…" | 1 | 1 | 0 |
| "== lile sex in silk == gay boy" | 1 | 1 | 0 |
| "do you think I give a flying fuck what some eurotrash asshole thinks?" | 1 | 1 | 0 |
| "Hey fuck you wikipedia! I do exist!" | 1 | 1 | 0 |

**Specialization (H1), independent of toxic-bert:** from §3.2, generic
`bert-base-classifier` (identity FPR 0.181) is *less* fair than specialized
`toxic-bert` (0.152) — the same direction as the SVM comparison, but here with
the confound of architecture family removed too (both are BERT-base-sized
encoders). This weakens a naive reading of H1 ("specialization costs fairness"):
if anything, *less* specialization was less fair in our results, suggesting
toxic-bert's larger-scale, higher-quality fine-tuning — not its specialization
per se — is what limits its bias, and likewise what gives it the robustness edge
in §3.3.

## 4. Conclusions and future work

1. **Toxicity detection is substantially lexical, but not purely so** — a
   50k-feature linear model reaches ~93% of a 109M encoder's F1 at ~3300× lower
   latency, yet the encoder retains a real, consistent edge at full scale;
   "bigger" buys a bounded, not a transformative, improvement.
2. **The encoder's more decisive value is robustness to obfuscation, not raw
   accuracy — and that robustness is not free with the architecture.** Our own
   `bert-base-classifier`, same architecture and parameter count as toxic-bert,
   is nearly as brittle to leetspeak as the linear SVM. toxic-bert's edge tracks
   its specific (larger-scale, in-domain) fine-tuning, not "being a BERT".
3. **Discriminative heads ≫ generative/zero-shot prompting, most cleanly shown
   on a single matched checkpoint** — `bert-base-classifier` dominates
   `bert-base-mlm-prompt` on every metric using the *identical* pretrained
   weights, isolating method from architecture. DistilGPT-2 prompting is
   near-chance and degenerate too, though BERT's zero-shot masked-LM prompt
   ranks meaningfully better (ROC-AUC 0.72) than GPT-2's causal one (≈0.4) —
   prompting method and base architecture both matter independently.
4. **Identity bias is present in every classifier and is data-driven, not tied
   to specialization** — our generic `bert-base-classifier` is *less* fair than
   the specialized toxic-bert, the opposite of a naive reading of H1. But the
   *mechanism* is not uniform: masking the identity token erases the bias for
   toxic-bert and the SVM (a clean lexical shortcut) but *increases* it for
   `bert-base-classifier`, whose over-flagging is evidently not reducible to a
   single token. Fixing bias via naive training-time masking (SVM) trades
   recall for little fairness gain.

**Future work.** Deliberately out of scope for this submission (no usable local
GPU, and/or beyond the time budget), rather than oversights:
- **hateBERT** as a cross-domain (Reddit-trained) specialized encoder.
- A **real distillation** run (toxic-bert → DistilBERT) on GPU/Colab, to isolate
  compression effects with a matched teacher/student pair.
- **Few-shot** decoder prompting (only zero-shot D1/D2 were run).
- **Calibration analysis** (ECE / reliability diagrams) for all five models.
- A **neural masked fine-tune** — our masking intervention (§3.4) retrains the
  identity-stripped model only for the TF-IDF+SVM; no transformer has been
  fine-tuned with identity terms masked.

## 5. References
- Jigsaw/Conversation AI. Toxic Comment Classification Challenge. Kaggle, 2018.
- Dixon, Li, Sorensen, Thain, Vasserman. Measuring and Mitigating Unintended Bias in Text Classification. AIES 2018.
- Hanu, Unitary. Detoxify / `unitary/toxic-bert`, 2020.
- Sanh et al. DistilBERT / DistilGPT-2, 2019.

## 6. Ethics statement (template — required)

**6.1 Introduction.** Students: Omer Ramon, Tom Jacob. Project: *Words That Harm —
architectures and fairness in toxic-comment detection.* We audit toxic-comment
classifiers for accuracy, identity bias, and robustness, to understand their
failure modes rather than to ship a moderation product.

**6.2 (LLM-assisted) stakeholders & explanations.**
- *Stakeholders:* (1) platform users/commenters, (2) human moderators, (3)
  identity groups at risk of over-flagging.
- *Explanations:* users need to know why a comment was flagged and how to appeal;
  moderators need per-label scores and confidence; identity groups need
  transparency about measured per-group false-positive rates.
- *Responsibility:* the platform deploying the model owns user- and
  group-facing explanations; the ML team owns moderator-facing documentation.

**6.3 Our reflection (WRITE OURSELVES — no LLM per the guidelines).**
[TODO Omer/Tom: what should change in the AI answers to be more ethical? e.g.
naming the identity-FPR disparities we measured, committing to human review for
identity-related flags, and publishing the bias audit.]

---

*Reproduce every figure: see `docs/PROGRESS.md` §5 and `docs/DEVELOPER_GUIDE.md`.*
