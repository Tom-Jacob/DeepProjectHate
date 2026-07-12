# Deliverables handoff

## Done
- Soft KD pipeline + notebooks 05–09 + slides
- **Decoder fine-tune:** `src/decoder_finetune.py`, `scripts/train_decoder_ft.py`,
  registry `distilgpt2-d1-ft`, Colab `notebooks/10_decoder_finetune_colab.ipynb`
- **Decoder Colab 10 weights** in `models/distilgpt2-d1-ft/` (flat HF layout;
  val answer acc 0.9415, 20k train / 2k val, 4 epochs)
- **§3.7 filled** from local eval (test n=200; fairness n=210):
  - zero-shot d1: F1 0.189, ROC-AUC 0.46, identity FPR 0.867
  - d1-ft: F1 **0.590**, ROC-AUC **0.965**, identity FPR **0.505** (control 0.0)
- **Distillation Colab 09 weights** in `models/distilbert-student/` + §3.6 numbers filled
  (test F1 0.772, identity FPR 0.062, 159 ms/comment)

## You still do
1. Write ethics §6.3 (human)
2. Record ≤6 min video from `report/Words_That_Harm_Slides.pptx`
3. Optional: re-export Word after number updates; refresh slides for §3.7
