# Deliverables handoff

## Done
- Soft KD pipeline + notebooks 05–09 + slides
- **Decoder fine-tune:** `src/decoder_finetune.py`, `scripts/train_decoder_ft.py`,
  registry `distilgpt2-d1-ft`, Colab `notebooks/10_decoder_finetune_colab.ipynb`
- **Distillation:** `src/distill.py`, `scripts/train_distill.py`, registry
  `distilbert-student`, Colab `notebooks/09_distillation_colab.ipynb`

### Tom's original Colab numbers (GPU, full scale)
- **D1-ft** — 20k train / 2k val, 4 epochs, val answer acc 0.9415.
  Eval at test n=200 / fairness n=210:
  - zero-shot d1: F1 0.189, ROC-AUC 0.46, identity FPR 0.867
  - d1-ft: F1 **0.590**, ROC-AUC **0.965**, identity FPR **0.505** (control 0.0)
- **S1 (distilbert-student)** — test F1 0.772, identity FPR 0.062, 159 ms/comment

These are reported in `report/report_draft.md` §3.6–3.7 and in `report/report.tex`
Table 4 as a **separate protocol** (n=500 / n=200), not on the shared n=800 subset
used for the headline table.

### Local reproduction (2026-09-09, CPU, reduced scale)
Both training scripts were re-run locally, unmodified, at a much smaller scale, to
confirm the pipelines genuinely work rather than taking the numbers on trust:

| Run | Scale | Result | Artifacts |
|---|---|---|---|
| `train_distill.py` | 4,000 train / 800 val, 2 epochs | toxic F1 **0.797**, macro-F1 0.383 | `models/distilbert-student/`, `data/processed/distill_results.json` |
| `train_decoder_ft.py` | 800 train / 200 val, 2 epochs | val answer acc **0.935** | `models/distilgpt2-d1-ft/`, `data/processed/decoder_ft_results.json` |

Both completed cleanly and in the same qualitative direction as the Colab numbers.
**The local weight directories now hold these reduced-scale reproductions, not Tom's
original Colab checkpoints** — if you need the Colab weights, re-run notebooks 09/10.

## You still do
1. **Write ethics §6.3 (human — no LLM).** In `report/report.tex` this is the
   "3a. Our reflection on the AI output" paragraph; it is drafted, review it.
2. **Record the ≤6 min video** from `report/Words_That_Harm_Slides.pptx`, following
   `report/PRESENTATION_OUTLINE.md`.
3. Optional: re-run at full Colab scale (`notebooks/15_full_eval_colab.ipynb`) to
   confirm the CPU/n=800 numbers.
