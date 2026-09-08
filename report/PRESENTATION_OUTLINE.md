# Presentation outline — Words That Harm (≤6 min / ~10 slides)

**Course:** EE046211 · **Authors:** Omer Ramon, Tom Jacob
**Repo:** https://github.com/Tom-Jacob/DeepProjectHate
**Figures:** `report/figures/`

> Rewritten to include the full current result set: the same-checkpoint
> classifier-vs-prompting experiment (E1/E1-p), the corrected (not-purely-lexical)
> accuracy finding, the divergent masking result, and Tom's distillation +
> decoder fine-tune. All numbers below are final, from the completed rerun
> (see `report/report_draft.md`). Bonus finding not yet in a slide: robustness
> is direction-dependent — toxic-bert resists evasion but is easily tricked
> into flagging neutral text with the same perturbations (FP-induction
> 65-91% vs. 11% baseline) — good Q&A material even without its own slide.
> Timing adds to ~6:05 — cut Slide 10 to bullet 1+2 only if you need the 5s back.

Use this outline to update `report/Words_That_Harm_Slides.pptx` (figures
already embedded there — swap in the regenerated PNGs after the rerun), then
record the video.

---

## Slide 1 — Title (10s)
**Words That Harm:** Architectures and Fairness in Toxic-Comment Detection
Omer Ramon · Tom Jacob · Technion EE046211

## Slide 2 — Motivation + hypotheses (35s)
- Moderation models hide/remove comments at scale; benchmark F1 hides two
  failure modes — missed hate **and** over-flagging identity mentions
- We **audit** models, not ship a product classifier
- **H1:** specialization trades fairness for accuracy
- **H2:** compression amplifies bias
- **H3:** LM prompting is weaker/format-sensitive than a trained head

## Slide 3 — Fair comparison setup (40s)
Same Jigsaw data, one interface `P(toxic)`, five model configurations:

| Model | Role |
|-------|------|
| toxic-bert (109M) | in-domain specialized encoder |
| **bert-base-classifier (109M)** | **generic encoder, fine-tuned by us** |
| **bert-base-mlm-prompt (109M)** | **same checkpoint, zero-shot prompted** |
| TF-IDF + LinearSVM | non-neural floor (we train) |
| DistilGPT-2 D1/D2 (+ D1-ft, Tom) | zero-shot decoder prompting + fine-tuned |

Also report **params + ms/comment** for every model.

## Slide 4 — Accuracy vs efficiency (40s)
*Show `model_comparison_bars_test.png` + `accuracy_latency_test.png`* (test, n=800)
- toxic-bert F1 0.752, PR-AUC 0.835 — the **SVM (F1 0.703, PR-AUC 0.784) comes
  close but does not match or beat it** at this sample size; still ~3300×
  cheaper (0.1ms vs. 175ms/comment)
- DistilGPT-2 near chance (ROC-AUC 0.39–0.46)
- Takeaway: toxicity detection is **substantially lexical, not purely so** —
  the encoder's extra capacity buys a real, bounded edge

## Slide 5 — Fairness / identity FPR (40s)
*Show `fairness_fpr_heatmap.png`* (368-sentence probe, 23 groups)
- Control FPR = 0 for every classifier → false positives are identity-driven
- toxic-bert FPR 0.087 > SVM FPR 0.049 → specialization costs fairness (**H1**)
- **But our own generic bert-base-classifier is FPR 0.103 — less fair than
  specialized toxic-bert, on the identical architecture.** Specialization
  alone doesn't explain the ranking (see Slide 8)
- Decoder control FPR ≈ 1.0 → degenerate

## Slide 6 — Robustness to obfuscation (35s)
*Show `robustness_evasion_heatmap.png`*
- toxic-bert baseline detection 0.93, leetspeak evasion **0.28**
  — far the hardest to evade
- SVM trivially evaded (leetspeak evasion **0.71** — features *are* tokens)
- **bert-base-classifier — same architecture/size as toxic-bert — is nearly as
  brittle (leetspeak evasion 0.72)**: the robustness edge tracks
  toxic-bert's specific fine-tuning, not "being a BERT"
- Bonus (Q&A): toxic-bert's robustness is direction-dependent — the same
  perturbations that barely evade a real toxic comment get a NEUTRAL comment
  wrongly flagged as toxic 65-91% of the time (vs. 11% baseline)

## Slide 7 — Masking: the shortcut, and the exception (40s)
*Show masking figures*
- Mask the identity token → FPR drops to **0.00** (toxic-bert)
  and **0.00** (SVM): a clean lexical shortcut
- **bert-base-classifier does the opposite — masking *increases* its FPR
  (0.10 → 0.19)**: the same data-driven bias, a
  different mechanism per model
- Training-time SVM masking doesn't fix FPR; costs identity-hate recall

## Slide 8 — Same checkpoint, two methods (45s) — headline slide
*Table from report §3.10*
- `bert-base-classifier` (trained head) vs. `bert-base-mlm-prompt` (zero-shot
  prompt) — **literally the same pretrained weights**, method is the only
  variable
- F1 0.675 vs. **0.000** — the trained head dominates
- But `bert-base-mlm-prompt`'s ROC-AUC (0.722) is far above DistilGPT-2's
  (≈0.4): it *can* rank comments, it's just miscalibrated at threshold 0.5
- **This is the cleanest test of H3 in the project** — architecture and
  pretraining held fixed, only method varies

## Slide 9 — Distillation + decoder fine-tune (Tom) (30s)
- Soft-KD DistilBERT student **matches** toxic-bert's F1 (0.772 vs. 0.770) at
  lower latency and **lower** identity FPR (0.062 vs. 0.152) — compression did
  not amplify bias (**H2 partially rejected**)
- Fine-tuning DistilGPT-2 on the D1 prompt recovers real discrimination
  (F1 0.19 → 0.59, ROC-AUC 0.46 → 0.97) — zero-shot failure is a supervision
  problem, not an architecture ceiling

## Slide 10 — Conclusions + ethics (35s)
1. Lexical floor is strong, but not the whole story
2. Encoder's edge is robustness — and it's earned, not free with the architecture
3. Discriminative head ≫ prompting, most cleanly shown same-checkpoint
4. Identity bias is data-driven but **not one universal mechanism**
5. Compression (soft-KD) didn't cost fairness here

**Ethics (say aloud):** we audit failure modes, not ship a moderation product;
over-flagging identity mentions needs human review and published per-group
FPR numbers — and since the *mechanism* differs per model (Slide 7), a fix
that works for one architecture won't automatically work for another.

---

## Video checklist
- [ ] Update `report/Words_That_Harm_Slides.pptx` with this content (a fresh,
      standalone draft deck with these numbers already applied exists as
      `Words_That_Harm_Slides_NEW.pptx`, outside the repo, for reference)
- [ ] Swap in the regenerated figures from `report/figures/`
- [ ] Keep total talk ≤ 6 minutes — Slide 10 down to bullets 1-2 if running long
- [ ] Mention the repo URL once
- [ ] Q&A prep: hateBERT and few-shot prompting are named future work, not
      oversights (see report §4); distillation/decoder-ft weights are Colab
      outputs not in the git repo (gitignored) — mention if asked about repro
