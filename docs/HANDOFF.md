# Deliverables handoff

## Done
- Soft KD pipeline + notebooks 05–09 + slides
- **Decoder fine-tune:** `src/decoder_finetune.py`, `scripts/train_decoder_ft.py`,
  registry `distilgpt2-d1-ft`, Colab `notebooks/10_decoder_finetune_colab.ipynb`,
  report §3.7 placeholders

## You still do (Colab → local)
1. Notebook **09** → `models/distilbert-student/` → fill §3.6
2. Notebook **10** → `models/distilgpt2-d1-ft/` → fill §3.7  
   ```bash
   python scripts/run_model_comparison.py --models toxic-bert distilgpt2-d1 distilgpt2-d1-ft --eval-n 200
   python scripts/run_fairness.py --models toxic-bert distilgpt2-d1 distilgpt2-d1-ft
   ```
3. Write ethics §6.3 (human)
4. Record ≤6 min video from `report/Words_That_Harm_Slides.pptx`
5. Optional: re-export Word after number updates
