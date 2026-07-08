# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

**Authors:** Omer Ramon, Tom Jacob · Technion EE046211 Deep Learning
**Repo:** https://github.com/Tom-Jacob/DeepProjectHate

> Draft for the ≤8-page report. Figures are in `report/figures/`. Numbers are the
> current runs (small samples where noted); re-run at full size before submission.
> The ethics section §6.3 must be written by us (no LLM) — see note there.

---

## Abstract

We audit toxic-comment detectors on the Jigsaw 2018 dataset across three
architectures evaluated on the **same data**: a specialized BERT encoder
(`unitary/toxic-bert`), a non-neural TF-IDF + linear-SVM baseline we train
ourselves, and a DistilGPT-2 decoder used as a zero-shot classifier via
prompting. Rather than chasing accuracy, we stress-test each model for
**identity bias**, **robustness to obfuscation**, and the **mechanism** behind
its errors. We find that (i) toxicity detection is largely lexical — a linear
model matches the encoder's F1 and beats its ranking quality at ~1700× lower
latency; (ii) the encoder's true advantage is robustness to obfuscation, not
accuracy; (iii) generative prompting with a small LM is near-chance and
degenerate; and (iv) all models share a data-driven identity bias that a masking
probe traces to a single-token lexical shortcut.

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

**Task & data.** Jigsaw 2018, binary target `toxic` (multi-label labels retained
for the encoder↔decoder analysis). Split 85/15 train/val from `train.csv`
(stratified), test = labeled `test.csv` rows (135,635 / 23,936 / 63,978).

**Architectures (same data, fair comparison).**
- **Encoder** — `unitary/toxic-bert` (BERT, 109M), fine-tuned on Jigsaw; 6 sigmoid heads, we read the `toxic` logit.
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
*Figures: `model_comparison_bars_test.png`, `accuracy_latency_test.png`.*

| model | params | F1 | PR-AUC | ROC-AUC | ms/comment |
|---|--:|:--:|:--:|:--:|--:|
| toxic-bert (BERT) | 109M | 0.755 | 0.817 | 0.974 | 310 |
| tfidf-svm (non-neural) | 50k | 0.750 | **0.887** | 0.975 | **0.18** |
| distilgpt2-d2 (decoder) | 82M | 0.235 | 0.110 | 0.41 | 668 |
| distilgpt2-d1 (decoder) | 82M | 0.205 | 0.209 | 0.47 | 743 |

The linear model **matches toxic-bert's F1 and beats its PR-AUC** at ~1700× lower
latency — most of the task is lexical. The decoder is near chance (ROC-AUC ≈ 0.4–0.5).
*(Test n=150, 20 positives — directional; re-run full before submission.)*

### 3.2 Fairness: per-identity FPR
*Figure: `fairness_fpr_heatmap.png`.*

| model | overall identity FPR | control FPR |
|---|:--:|:--:|
| tfidf-svm | 0.086 | 0.000 |
| toxic-bert | 0.152 | 0.000 |
| distilgpt2-d1 | 0.867 | **1.000** |

Control FPR is 0 for the classifiers → false positives are *specifically*
identity-driven ("I am gay", "I am Muslim", "I am transgender" are flagged). The
specialized encoder is **less fair** than the linear baseline (H1). The decoder
flags essentially everything (control FPR 1.0) — degenerate as a classifier.

### 3.3 Robustness to obfuscation
*Figure: `robustness_evasion_heatmap.png`.* Evasion = flagged-toxic comment that
slips through after perturbation (higher = more brittle).

| model | baseline det. | leetspeak | space-out | punct-noise |
|---|:--:|:--:|:--:|:--:|
| toxic-bert | 0.93 | **0.28** | 0.05 | 0.15 |
| tfidf-svm | 0.76 | 0.71 | 0.75 | 0.75 |
| distilgpt2-d1 | 0.58 | 0.10* | 0.17* | 0.17* |

The encoder is far the most robust; the SVM is trivially evaded (its features
*are* surface tokens). The decoder's low evasion is an artifact of near-random,
insensitive predictions (baseline detection only 0.58), not genuine robustness.
No model is fooled by ALL-CAPS.

### 3.4 Masking intervention (why they over-flag identities)
*Figures: `masking_inference_ablation.png`, `masking_svm_tradeoff.png`.*

Inference masking (replace the identity term with `[MASK]`): identity FPR
**0.152 → 0.000** (toxic-bert) and **0.086 → 0.000** (SVM). Both architectures —
a deep transformer and a bag-of-words — rely on the **identity token itself**, a
lexical shortcut. Training-time masking of the SVM does *not* reduce FPR (0.086 →
0.10) but costs identity-hate recall (0.78 → 0.68): naive removal is an
insufficient and harmful "fix".

### 3.5 Encoder vs. decoder agreement
*Figures: `enc_dec_gold_agreement.png`, `enc_dec_agreement.png`. (test n=150, 6 labels.)*

Overall encoder↔decoder agreement is only **0.187** (per-label 0.16–0.23). Against
gold on `toxic`, the encoder scores **0.913** but the decoder only **0.227** — the
two barely agree not because they capture different *valid* signals, but because
the decoder is degenerate: it labels almost every comment toxic. Of 150 comments
the two disagree on `toxic` **115** times, and the disagreements are overwhelmingly
decoder false positives on benign text.

| comment (truncated) | gold | encoder | decoder |
|---|:--:|:--:|:--:|
| ":Replied here:" | 0 | 0 | 1 |
| "US New Citizenship Test. In October of 2008 …" | 0 | 0 | 1 |
| "This is your last warning. You will be blocked …" | 0 | 0 | 1 |

Takeaway: cross-architecture agreement is a poor proxy for correctness here — a
strong discriminative encoder and a degenerate generative prompt "disagree" almost
everywhere, and the encoder is right nearly every time.

## 4. Conclusions and future work

1. **Toxicity detection is mostly lexical** — a 50k-feature linear model rivals a
   109M encoder; "bigger" is not the story.
2. **The encoder's real value is robustness to obfuscation**, not raw accuracy.
3. **Discriminative heads ≫ generative prompting at small scale** — DistilGPT-2
   prompting is near-chance, degenerate, and slowest; prompt format (D1 vs D2)
   changes results.
4. **Identity bias is data-driven, not architectural** — it appears even in the
   linear model, and masking the identity token erases it, revealing a lexical
   shortcut. Fixing it via naive training-masking trades recall for little gain.

**Future work.** Our own controlled distillation (toxic-bert → DistilBERT on
GPU/Colab) to isolate compression effects; a neural masked fine-tune;
calibration analysis; larger evaluation samples.

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
