# Deliverables handoff

## Done
- Soft KD pipeline + notebooks 05–09 + slides
- **Decoder fine-tune:** `src/decoder_finetune.py`, `scripts/train_decoder_ft.py`,
  registry `distilgpt2-d1-ft`, Colab `notebooks/10_decoder_finetune_colab.ipynb`,
  report §3.7 placeholders
- **Distillation Colab 09 weights** in `models/distilbert-student/` + §3.6 numbers filled
  (test F1 0.772, identity FPR 0.062, 159 ms/comment)

## You still do (Colab → local)
1. Notebook **10** → `models/distilgpt2-d1-ft/` → fill §3.7  
   ```bash
   python scripts/run_model_comparison.py --models toxic-bert distilgpt2-d1 distilgpt2-d1-ft --eval-n 200
   python scripts/run_fairness.py --models toxic-bert distilgpt2-d1 distilgpt2-d1-ft
   ```
2. Write ethics §6.3 (human)
3. Record ≤6 min video from `report/Words_That_Harm_Slides.pptx`
4. Optional: re-export Word after number updates
