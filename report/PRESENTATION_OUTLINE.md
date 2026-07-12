# Presentation outline — Words That Harm (≤6 min / ~10 slides)

**Course:** EE046211 · **Authors:** Omer Ramon, Tom Jacob  
**Repo:** https://github.com/Tom-Jacob/DeepProjectHate  
**Figures:** `report/figures/`

Use this outline to build PowerPoint / Google Slides, then record the video.

---

## Slide 1 — Title (15s)
**Words That Harm:** Architectures and Fairness in Toxic-Comment Detection  
Omer Ramon · Tom Jacob · Technion EE046211

## Slide 2 — Motivation (30s)
- Moderation models hide / remove comments at scale
- Benchmark F1 hides two failure modes: missed hate **and** over-flagging identity mentions
- We **audit** models — not ship a product classifier

## Slide 3 — Research questions (30s)
- **H1:** Specialized encoder (toxic-bert) trades fairness for accuracy
- **H2:** Smaller / compressed models lose accuracy *and* amplify bias
- **H3:** Decoder prompting (DistilGPT-2) is weaker / format-sensitive vs discriminative heads

## Slide 4 — Fair comparison setup (40s)
Three architectures, **same Jigsaw data**, one interface `P(toxic)`:
| Model | Role |
|-------|------|
| toxic-bert (109M) | in-domain encoder |
| TF-IDF + LinearSVM | non-neural floor (we train) |
| DistilGPT-2 D1/D2 | zero-shot decoder prompting |

Also report **params + ms/comment**.

## Slide 5 — Accuracy vs efficiency (45s)
*Show `model_comparison_bars_test.png` + `accuracy_latency_test.png`*
- SVM ≈ toxic-bert F1; **beats** PR-AUC at ~1700× lower latency
- DistilGPT-2 near chance (ROC-AUC ≈ 0.4–0.5)
- Takeaway: toxicity detection is largely **lexical**

## Slide 6 — Fairness / identity FPR (45s)
*Show `fairness_fpr_heatmap.png`*
- Control FPR = 0 for classifiers → FPs are identity-driven
- toxic-bert FPR > SVM FPR → specialization costs fairness (**H1**)
- Decoder control FPR ≈ 1.0 → degenerate classifier

## Slide 7 — Robustness to obfuscation (40s)
*Show `robustness_evasion_heatmap.png`*
- toxic-bert much harder to evade (leetspeak)
- SVM trivially evaded (features *are* tokens)
- Encoder's real edge is **robustness**, not raw F1

## Slide 8 — Masking intervention (40s)
*Show masking figures*
- Mask identity token → FPR drops to **0** (encoder and SVM)
- Bias is a **lexical shortcut**, not architecture-specific
- Training-time SVM masking does **not** fix FPR; hurts identity-hate recall

## Slide 9 — Encoder vs decoder (40s)
*Show `enc_dec_*.png`*
- Agreement ~0.19; decoder labels almost everything toxic
- Disagreements are mostly decoder false positives
- At small scale: discriminative head ≫ generative prompting (**H3**)

## Slide 10 — Conclusions + ethics (40s)
1. Lexical floor is strong  
2. Encoder value = robustness  
3. Small LM prompting fails here  
4. Identity bias is data-driven / token-level  

**Ethics (say aloud):** we audit failure modes; over-flagging identity mentions needs human review and published bias numbers.  
**Future:** fill distillation numbers after Colab notebook 09; larger eval samples.

---

## Video checklist
- [ ] Open `report/Words_That_Harm_Slides.pptx` (figures already embedded)
- [ ] Keep total talk ≤ 6 minutes
- [ ] Mention repo URL once
- [ ] Q&A prep: distillation = soft KD 6-label DistilBERT (Colab); hateBERT deferred
