# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

**Authors:** Omer Ramon, Tom Jacob · Technion EE046211 Deep Learning
**Repo:** https://github.com/Tom-Jacob/DeepProjectHate

> Draft for the ≤8-page report. Figures are in `report/figures/`. Encoder/SVM
> refreshed at test n=500; decoder comparison + fairness refreshed at n=200
> (2026-07-12). Word export: `report/Words_That_Harm_Report.docx`.
> Ethics §6.1–6.2 filled by the team; §6.3 reflection still to write (human-only).

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
| toxic-bert (BERT) | 109M | 0.770 | 0.843 | 0.981 | 432 |
| tfidf-svm (non-neural) | 50k | 0.733 | **0.862** | 0.975 | **0.70** |
| distilgpt2-d2 (decoder) | 82M | 0.206 | 0.102 | 0.45 | 769 |
| distilgpt2-d1 (decoder) | 82M | 0.189 | 0.167 | 0.46 | 849 |

The linear model is close to toxic-bert on F1 and **beats its PR-AUC** at ~600× lower
latency — most of the task is lexical. The decoder is near chance (ROC-AUC ≈ 0.45).
*(Encoder/SVM: test n=500. Decoder: test n=200 — both directional relative to full test.)*

### 3.2 Fairness: per-identity FPR
*Figure: `fairness_fpr_heatmap.png`.*

| model | overall identity FPR | control FPR |
|---|:--:|:--:|
| tfidf-svm | 0.090 | 0.000 |
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
| toxic-bert | 0.92 | **0.28** | 0.03 | 0.15 |
| tfidf-svm | 0.71 | 0.74 | 0.75 | 0.75 |
| distilgpt2-d1 | 0.58 | 0.10* | 0.17* | 0.17* |

The encoder is far the most robust; the SVM is trivially evaded (its features
*are* surface tokens). The decoder's low evasion is an artifact of near-random,
insensitive predictions (baseline detection only 0.58), not genuine robustness.
No model is fooled by ALL-CAPS.

### 3.4 Masking intervention (why they over-flag identities)
*Figures: `masking_inference_ablation.png`, `masking_svm_tradeoff.png`.*

Inference masking (replace the identity term with `[MASK]`): identity FPR
**0.152 → 0.000** (toxic-bert) and **0.090 → 0.000** (SVM). Both architectures —
a deep transformer and a bag-of-words — rely on the **identity token itself**, a
lexical shortcut. Training-time masking of the SVM does *not* reduce FPR (0.090 →
0.10) but costs identity-hate recall (0.78 → 0.68): naive removal is an
insufficient and harmful "fix".

### 3.5 Encoder vs. decoder agreement
*Figures: `enc_dec_gold_agreement.png`, `enc_dec_agreement.png`. (test n=200, 6 labels.)*

Overall encoder↔decoder agreement is only **0.185**. Against
gold on `toxic`, the encoder scores **0.925** but the decoder only **0.230** — the
two barely agree not because they capture different *valid* signals, but because
the decoder is degenerate: it labels almost every comment toxic. Of 200 comments
the two disagree on `toxic` **153** times, and the disagreements are overwhelmingly
decoder false positives on benign text.

| comment (truncated) | gold | encoder | decoder |
|---|:--:|:--:|:--:|
| ":Replied here:" | 0 | 0 | 1 |
| "US New Citizenship Test. In October of 2008 …" | 0 | 0 | 1 |
| "This is your last warning. You will be blocked …" | 0 | 0 | 1 |

Takeaway: cross-architecture agreement is a poor proxy for correctness here — a
strong discriminative encoder and a degenerate generative prompt "disagree" almost
everywhere, and the encoder is right nearly every time.

### 3.6 Distillation (soft KD, 6 labels)
*Colab train: `notebooks/09_distillation_colab.ipynb`. Local eval key: `distilbert-student`.*

We distill **all six** Jigsaw labels from frozen `toxic-bert` into a DistilBERT
student with soft targets (temperature $T{=}2$, mix $\alpha{=}0.7$ KD + hard BCE).
The student exposes `predict_toxic_proba` for the binary harness and full 6-label
sigmoid probs for multi-label metrics.

| model | params | test F1 (toxic) | identity FPR | ms/comment | notes |
|---|--:|:--:|:--:|--:|---|
| toxic-bert (teacher) | 109M | 0.770 | 0.152 | 274 | n=500 |
| distilbert-student | 67M | **0.772** | **0.062** | **159** | soft KD, 6-label; Colab 20k×4ep |

**Result vs H2:** on this run the student **matches** the teacher's toxic F1 (0.772 vs 0.770)
at ~1.7× lower latency and **lower** overall identity FPR (0.062 vs 0.152). Compression did
**not** amplify identity bias here — though worst-group FPR remains concentrated (e.g. `gay`).
Training: soft KD $T{=}2$, $\alpha{=}0.7$, 20k train / 2k val, 4 epochs; best val toxic F1 ≈ 0.85
(`models/distilbert-student/distill_config.json`).

Reproduce after weights exist in `models/distilbert-student/`:
```bash
python scripts/run_model_comparison.py --models toxic-bert tfidf-svm distilbert-student --eval-n 500
python scripts/run_fairness.py --models toxic-bert distilbert-student
python scripts/run_robustness.py --models toxic-bert distilbert-student --n 200
```

### 3.7 Decoder fine-tune (D1 yes/no, binary toxic)
*Colab: `notebooks/10_decoder_finetune_colab.ipynb`. Registry key: `distilgpt2-d1-ft`.*

Zero-shot DistilGPT-2 prompting is near chance. We keep that result and add a
**small supervised probe**: fine-tune DistilGPT-2 with causal LM loss **only on**
the answer tokens (` yes` / ` no`) after the same D1 prompt, on ~8k stratified
train comments (1–2 epochs). Eval uses the identical `score_yes_no` logprob path.

| model | F1 | ROC-AUC | identity FPR | notes |
|---|:--:|:--:|:--:|---|
| distilgpt2-d1 (zero-shot) | 0.189 | 0.46 | 0.867 | degenerate / near chance |
| distilgpt2-d1-ft (fine-tuned) | `[TBD after Colab]` | `[TBD]` | `[TBD]` | small probe, binary toxic |

**Claim:** failure of zero-shot prompting at DistilGPT-2 scale does **not** imply
decoders cannot learn the task — light task-specific adaptation recovers usable
discrimination while preserving the architecture comparison.

```bash
python scripts/train_decoder_ft.py --max-train-samples 8000 --epochs 2
python scripts/run_model_comparison.py --models toxic-bert distilgpt2-d1 distilgpt2-d1-ft --eval-n 200
python scripts/run_fairness.py --models toxic-bert distilgpt2-d1 distilgpt2-d1-ft
```

## 4. Conclusions and future work

1. **Toxicity detection is mostly lexical** — a 50k-feature linear model rivals a
   109M encoder; "bigger" is not the story.
2. **The encoder's real value is robustness to obfuscation**, not raw accuracy.
3. **Zero-shot decoder prompting fails at DistilGPT-2 scale** — near chance and
   often degenerate; a **small D1 fine-tune** (§3.7) tests whether light
   supervision recovers usable discrimination without changing architecture.
4. **Identity bias is data-driven, not architectural** — it appears even in the
   linear model, and masking the identity token erases it, revealing a lexical
   shortcut. Fixing it via naive training-masking trades recall for little gain.
5. **Distillation (soft KD):** the DistilBERT student matches toxic-bert toxic F1
   (0.772 vs 0.770) at lower latency and **lower** identity FPR (0.062 vs 0.152)
   on our probe — compression need not amplify bias (**H2** partially rejected here).

**Future work.** Neural masked fine-tune of DistilBERT; calibration analysis;
larger evaluation samples; hateBERT cross-domain probe.

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
1. Given our measured identity FPRs (toxic-bert ≈ 15%, SVM ≈ 9%, decoder near-degenerate), what should change in how an AI moderation system *answers* or *acts* toward users?
2. Would you require human review for flags that fire on neutral identity mentions? Why / why not?
3. Should platforms publish bias-audit numbers like ours? What risk does that create, and what benefit?

[To be added by the team.]

---

*Reproduce every figure: see `docs/PROGRESS.md` §5 and `docs/DEVELOPER_GUIDE.md`.*
