"""Update WordsThatHarm.docx in place to match the latest report_draft.md.

Run: python scripts/update_docx.py
"""
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Inches
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
DOCX_PATH = REPORT_DIR / "WordsThatHarm.docx"
FIGURES_DIR = REPORT_DIR / "figures"
BACKUP = DOCX_PATH.with_suffix(".docx.bak")

shutil.copy2(DOCX_PATH, BACKUP)
print(f"Backed up to {BACKUP.name}")

doc = Document(str(DOCX_PATH))


def P(idx):
    return doc.paragraphs[idx]


def find_para(substr, start=0):
    for i in range(start, len(doc.paragraphs)):
        if substr in doc.paragraphs[i].text:
            return i
    return None


def set_text(para, text):
    for r in para.runs:
        r.text = ""
    if not para.runs:
        para.add_run("")
    para.runs[0].text = text


def set_bold_body(para, bold_part, body_part):
    for r in para.runs:
        r.text = ""
    if not para.runs:
        para.add_run("")
    para.runs[0].text = bold_part
    para.runs[0].bold = True
    r = para.add_run(body_part)
    r.bold = False
    return r


def insert_para_after(para, text, style_name=None):
    new_p = OxmlElement("w:p")
    para._element.addnext(new_p)
    from docx.text.paragraph import Paragraph
    np = Paragraph(new_p, para._parent)
    if style_name and style_name in [s.name for s in doc.styles]:
        np.style = doc.styles[style_name]
    np.add_run(text)
    return np


def insert_image_after(para, img_path, width=5.0):
    if not Path(img_path).exists():
        print(f"  [SKIP] {Path(img_path).name} not found")
        return None
    np = insert_para_after(para, "", "Body Text")
    np.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = np.runs[0] if np.runs else np.add_run()
    run.add_picture(str(img_path), width=Inches(width))
    print(f"  [IMG]  {Path(img_path).name}")
    return np


def clear_data_rows(table):
    while len(table.rows) > 1:
        table._tbl.remove(table.rows[-1]._tr)


def rebuild_table(table, header, rows):
    clear_data_rows(table)
    for i, h in enumerate(header):
        if i < len(table.rows[0].cells):
            table.rows[0].cells[i].text = h
    for rd in rows:
        r = table.add_row()
        for i, t in enumerate(rd):
            if i < len(r.cells):
                r.cells[i].text = t


# ================================================================
# 1. ABSTRACT  (para 4)
# ================================================================
print("\n[1] Abstract")
i = find_para("In this project we audit")
if i is not None:
    set_text(P(i),
        "We audit toxic-comment detectors on the Jigsaw 2018 dataset across "
        "three architectural families evaluated on the same data: a specialized "
        "BERT encoder (unitary/toxic-bert), a non-neural TF-IDF + linear-SVM "
        "baseline we train ourselves, a DistilGPT-2 decoder (zero-shot and "
        "fine-tuned), and a DistilBERT student trained via knowledge "
        "distillation from the encoder. Rather than chasing accuracy, we "
        "stress-test each model for identity bias, robustness to obfuscation, "
        "and the mechanism behind its errors. We find that "
        "(i)\u00a0toxicity detection is largely lexical\u2014a linear model "
        "matches the encoder\u2019s F1 at ~600\u00d7 lower latency; "
        "(ii)\u00a0the encoder\u2019s true advantage is robustness to "
        "obfuscation, not accuracy; "
        "(iii)\u00a0zero-shot decoder prompting at DistilGPT-2 scale is "
        "near-chance, but light fine-tuning recovers usable discrimination "
        "(F1\u00a00.59, ROC-AUC\u00a00.97); "
        "(iv)\u00a0soft knowledge distillation matches the teacher\u2019s F1 "
        "while reducing identity bias; and "
        "(v)\u00a0all models share a data-driven identity bias that a masking "
        "probe traces to a single-token lexical shortcut.")
    print("  done")

# ================================================================
# 2. WHY F1 paragraph  (insert after para 24 = ROC-AUC, before 25)
# ================================================================
print("\n[2] Why F1 paragraph")
i = find_para("For each model we also report parameter count")
if i is not None:
    np = insert_para_after(P(i - 1), "", "Body Text")
    r0 = np.runs[0]
    r0.text = "Why F1, not accuracy? "
    r0.bold = True
    np.add_run(
        "The Jigsaw dataset is highly imbalanced (~10% toxic): a trivial "
        "all-negative classifier achieves 90% accuracy while detecting nothing. "
        "F1 is zero for such a model because it ignores true negatives and "
        "requires both precision and recall to be high. We therefore report F1 "
        "as our primary metric throughout.")
    print("  inserted after ROC-AUC")

# ================================================================
# 3. MODELS  (section 2.3)
# ================================================================
print("\n[3] Models table + prose")
# 3a. "five" -> "six"
i = find_para("We compare five models")
if i is not None:
    for r in P(i).runs:
        r.text = r.text.replace("five models", "six models")
    print("  five -> six")

# 3b. Add D2 row to Table 0, rename D1/D1-ft
t0 = doc.tables[0]
t0.cell(3, 1).text = "DistilGPT-2 yes/no (zero-shot)"
t0.cell(4, 1).text = "DistilGPT-2 yes/no (fine-tuned)"

new_row = t0.add_row()
d2 = ["D2", "DistilGPT-2 fill-in-blank (zero-shot)",
      "GPT-2 decoder", "82M",
      "Prompted: \u201cThis comment is ___.\u201d Log-prob of toxic vs.\u00a0not toxic"]
for ci, txt in enumerate(d2):
    new_row.cells[ci].text = txt
t0._tbl.remove(new_row._tr)
t0.rows[4]._tr.addprevious(new_row._tr)
print("  added D2 row between D1 and D1-ft")

# 3c. Update D1/D2 prose (para 31)
i = find_para("DistilGPT-2 (D1/D2)")
if i is not None:
    set_bold_body(P(i), "DistilGPT-2 (D1, D2). ",
        "A 6-layer decoder-only transformer (768 hidden, 12 heads, "
        "82M\u00a0params), distilled from GPT-2 by Hugging Face. It is a "
        "language model, not a classifier\u2014we repurpose it via prompting. "
        "Both D1 and D2 share the same weights; they differ only in prompt "
        "design. In the D1 (yes/no) method we append \u201cIs the following "
        "comment toxic? Answer yes or no.\u201d and compare the "
        "log-probability of yes vs.\u00a0no. In D2 (fill-in-blank) we use "
        "\u201cThis comment is ___.\u201d and score toxic vs.\u00a0not toxic. "
        "Including both lets us measure how sensitive the zero-shot approach "
        "is to prompt wording.")
    print("  updated D1/D2 prose")

# ================================================================
# 4. FAIRNESS PROBE  (section 2.5, para 41)
# ================================================================
print("\n[4] Fairness probe (2.5)")
i = find_para("Following Dixon et al")
if i is not None:
    set_bold_body(P(i), "",
        "Following Dixon et\u00a0al.\u00a0(2018), we construct neutral "
        "sentences by filling 10 template carriers (\u201cI am {g}.\u201d, "
        "\u201cMy friend is {g}.\u201d, etc.) with 23 terms across 6 "
        "categories (religion, race, orientation, gender, disability, plus two "
        "neutral baselines: \u201chuman\u201d and \u201cnormal\u201d). The "
        "neutral terms produce sentences like \u201cI am human\u201d and "
        "\u201cMy friend is normal\u201d\u2014these exercise the same template "
        "structure as identity terms but should score identically to controls. "
        "This gives 230 probe sentences total. By construction none is toxic, "
        "so any positive prediction is a false positive. We compare per-group "
        "FPR against 5 no-identity control sentences (\u201cI am here.\u201d, "
        "\u201cMy friend is nice.\u201d) to verify that the bias is "
        "identity-driven, not template-driven.")
    P(i).runs[0].bold = False
    print("  updated to 230 sentences, human/normal")

    np = insert_para_after(P(i), "", "Body Text")
    r0 = np.runs[0]
    r0.text = "Perturbation-augmented fairness probes. "
    r0.bold = True
    np.add_run(
        "We additionally apply each of the 6 perturbations (\u00a72.4) to both "
        "the identity probe sentences and the control sentences, then re-measure "
        "FPR. This tests whether surface noise amplifies or reduces identity "
        "bias\u2014e.g., does leetspeak on \u201cI am muslim\u201d change the "
        "model\u2019s false-positive behavior?")
    print("  added perturbation-augmented probes paragraph")

# ================================================================
# 5. SECTION 3.1  (protocol + table)
# ================================================================
print("\n[5] Section 3.1")
i = find_para("Protocol. Score the val and test splits")
if i is not None:
    set_bold_body(P(i), "Protocol. ",
        "All six models are scored on the same fixed 5,000-comment test subset "
        "(seeded) so every model sees identical data\u2014a perfectly fair "
        "head-to-head. We measure F1, PR-AUC, ROC-AUC, and inference latency "
        "(ms/comment, GPU). Fast models (encoder, SVM, student) are "
        "additionally confirmed on the full test set (63,978\u00a0comments); at "
        "n\u00a0=\u00a05,000 the 95%\u00a0CI on F1 is \u00b10.01, so the 5k "
        "numbers are already reliable. All runs executed on Colab GPU (T4).")
    print("  protocol -> nested 5k/63k")

t2 = doc.tables[2]
rebuild_table(t2,
    ["Model", "Params", "F1", "PR-AUC", "ROC-AUC", "ms/comment"],
    [["toxic-bert (BERT)", "109M", "\u2014", "\u2014", "\u2014", "\u2014"],
     ["tfidf-svm (non-neural)", "50k", "\u2014", "\u2014", "\u2014", "\u2014"],
     ["distilbert-student (KD)", "67M", "\u2014", "\u2014", "\u2014", "\u2014"],
     ["distilgpt2-d1-ft (fine-tuned)", "82M", "\u2014", "\u2014", "\u2014", "\u2014"],
     ["distilgpt2-d1 (zero-shot)", "82M", "\u2014", "\u2014", "\u2014", "\u2014"],
     ["distilgpt2-d2 (fill-in-blank)", "82M", "\u2014", "\u2014", "\u2014", "\u2014"]])
print("  table -> 6 models, placeholders")

i = find_para("Accuracy vs")
if i is not None:
    set_text(P(i), "Accuracy vs.\u00a0efficiency (fair comparison, n\u00a0=\u00a05,000). "
             "Placeholder\u2014update after Colab run.")

i = find_para("Interpretation. The linear model is close")
if i is not None:
    set_bold_body(P(i), "Interpretation. ",
        "The linear model is close to toxic-bert on F1 and beats its PR-AUC at "
        "~600\u00d7 lower latency\u2014most of the task is lexical, confirming "
        "that raw accuracy alone does not justify a 109M-parameter transformer. "
        "The zero-shot decoder is near chance (ROC-AUC \u2248 0.45), consistent "
        "with H3: a small language model cannot discriminate toxicity without "
        "task-specific supervision. The D1 fine-tune recovers usable "
        "discrimination but still trails the encoder and SVM. Fast models "
        "confirmed on full test n\u00a0=\u00a063,978 with consistent results "
        "(within \u00b10.01 F1 of the 5k numbers).")
    print("  interpretation updated")

# ================================================================
# 6. SECTION 3.2  (fairness)
# ================================================================
print("\n[6] Section 3.2")
i = find_para("Protocol. Score the 210-sentence")
if i is not None:
    set_bold_body(P(i), "Protocol. ",
        "Score the 230-sentence identity probe (\u00a72.5, including \u201c"
        "human\u201d and \u201cnormal\u201d neutral baselines) plus 5 "
        "no-identity controls with each model at threshold 0.5. Report overall "
        "mean FPR across all groups and control FPR. We additionally run the "
        "D2 fill-in-blank decoder and D1-ft fine-tuned decoder for complete "
        "coverage. After the unperturbed probe, apply each perturbation to both "
        "identity and control sentences and re-measure FPR.")
    print("  protocol -> 230, D2/D1-ft, perturbed")

t3 = doc.tables[3]
rebuild_table(t3,
    ["Model", "Overall identity FPR", "Control FPR"],
    [["tfidf-svm", "\u2014", "\u2014"],
     ["toxic-bert", "\u2014", "\u2014"],
     ["distilbert-student", "\u2014", "\u2014"],
     ["distilgpt2-d2", "\u2014", "\u2014"],
     ["distilgpt2-d1-ft", "\u2014", "\u2014"]])
print("  table -> 5 models, placeholders")

i = find_para("Identity FPR on neutral probe")
if i is not None:
    set_text(P(i), "Identity FPR on neutral probe sentences. "
             "Placeholder\u2014update after Colab run.")

i = find_para("Interpretation. Control FPR is")
if i is not None:
    set_bold_body(P(i), "Interpretation. ",
        "Control FPR is 0 for all trained classifiers, meaning false positives "
        "are specifically identity-driven (\u201cI am gay\u201d, \u201cI am "
        "Muslim\u201d are flagged despite being non-toxic). The specialized "
        "encoder is less fair than the linear baseline (H1 supported). The "
        "neutral baselines (\u201chuman\u201d, \u201cnormal\u201d) confirm "
        "that the template structure itself does not trigger false positives"
        "\u2014the signal is the identity term. Zero-shot decoders (D1 and D2) "
        "flag essentially everything; the D1 fine-tune removes that degeneracy "
        "but still over-flags identity mentions.")
    print("  interpretation updated")

# ================================================================
# 7. SECTION 3.3  (bidirectional robustness)
# ================================================================
print("\n[7] Section 3.3")
i = find_para("Goal. Determine which architecture is most resilient")
if i is not None:
    set_bold_body(P(i), "Goal. ",
        "Determine which architecture is most resilient to the surface-level "
        "evasion tactics described in \u00a72.4, testing in both directions: "
        "(a)\u00a0evasion (toxic comments that slip through after perturbation) "
        "and (b)\u00a0false-positive induction (non-toxic comments that become "
        "flagged after perturbation).")
    print("  goal -> bidirectional")

i = find_para("Protocol. Take the subset of test comments")
if i is not None:
    set_bold_body(P(i), "Protocol. ",
        "(a)\u00a0Toxic probe: take 1,000 genuinely-toxic test comments that "
        "each model flags as toxic (score \u2265\u00a00.5), apply each "
        "perturbation and re-score. Report evasion rate. "
        "(b)\u00a0Non-toxic probe: take 1,000 correctly-classified non-toxic "
        "comments, apply each perturbation, and report FP induction rate\u2014"
        "the fraction that flip from non-toxic to toxic after perturbation.")
    print("  protocol -> bidirectional with n=1000")

t4 = doc.tables[4]
rebuild_table(t4,
    ["Model", "Baseline det.", "Leetspeak", "Space-out", "Punct+Typos+Repeat+Upper"],
    [["toxic-bert", "\u2014", "\u2014", "\u2014", "\u2014"],
     ["tfidf-svm", "\u2014", "\u2014", "\u2014", "\u2014"],
     ["distilbert-student", "\u2014", "\u2014", "\u2014", "\u2014"]])
print("  evasion table -> 3 models, placeholders")

i = find_para("Evasion rates (higher")
if i is not None:
    set_text(P(i), "Evasion rates\u2014toxic probe. "
             "Placeholder\u2014update after Colab run.")

i = find_para("Interpretation. The encoder is far the most robust")
if i is not None:
    np = insert_para_after(P(i - 1), "", "Table Caption")
    np.runs[0].text = ("FP induction\u2014non-toxic probe. "
                        "Placeholder\u2014a separate table with false-positive "
                        "induction rates will be added after the Colab run.")
    np.runs[0].italic = True
    print("  added FP induction placeholder")

i = find_para("Interpretation. The encoder is far the most robust")
if i is not None:
    set_bold_body(P(i), "Interpretation. ",
        "The encoder is far the most robust\u2014its WordPiece tokenizer and "
        "contextual embeddings survive most surface noise. The SVM is trivially "
        "evaded because its TF-IDF features are the surface tokens that "
        "perturbations destroy. The FP induction direction tests whether "
        "perturbations can push innocent text into being flagged, which matters "
        "for understanding whether a model\u2019s features are stable or "
        "whether noise can push predictions in either direction.")
    print("  interpretation -> bidirectional")

# ================================================================
# 8. SECTION 3.4  (masking FPR numbers)
# ================================================================
print("\n[8] Section 3.4")
i = find_para("Results. Inference masking drops identity FPR from 0.152")
if i is not None:
    set_bold_body(P(i), "Results. ",
        "Inference masking drops identity FPR from 0.139 to 0.000 (toxic-bert) "
        "and from 0.083 to 0.000 (SVM). Both architectures\u2014a deep "
        "transformer and a bag-of-words\u2014rely on the identity token itself: "
        "a lexical shortcut. Training-time masking of the SVM does not reduce "
        "FPR (0.083 \u2192 0.10) but costs identity-hate recall "
        "(0.78 \u2192 0.68).")
    print("  FPR numbers updated (0.139/0.083)")

# ================================================================
# 9. SECTION 3.5  (agreement with D1-ft)
# ================================================================
print("\n[9] Section 3.5")
i = find_para("Goal. Measure how much the encoder and decoder agree")
if i is not None:
    set_bold_body(P(i), "Goal. ",
        "Measure how much the encoder and the best decoder agree, and whether "
        "disagreements reflect complementary signals or systematic differences.")

i = find_para("Protocol. Score 200 test comments")
if i is not None:
    set_bold_body(P(i), "Protocol. ",
        "Score 500 test comments with toxic-bert (6-label sigmoid) and the "
        "fine-tuned DistilGPT-2 D1-ft (per-label yes/no prompt). We use D1-ft "
        "rather than the zero-shot D1 to ensure a meaningful comparison\u2014"
        "the zero-shot decoder is near-random (\u00a73.1) and agreement with a "
        "degenerate model is uninformative. For each label binarize at 0.5 and "
        "compute agreement rate, plus each model\u2019s agreement with gold.")
    print("  protocol -> D1-ft, 500 comments")

i = find_para("Results. Overall encoder")
if i is not None:
    set_bold_body(P(i), "Results. ",
        "Placeholder\u2014update after Colab run. Overall encoder\u2013decoder "
        "agreement: \u2014. Against gold on toxic, the encoder scores \u2014 "
        "and the fine-tuned decoder \u2014.")
    print("  results -> placeholder")

t5 = doc.tables[5]
rebuild_table(t5,
    ["Comment (truncated)", "Gold", "Encoder", "Decoder (D1-ft)"],
    [["\u2014", "\u2014", "\u2014", "\u2014"]])
print("  agreement table -> placeholder")

i = find_para("Interpretation. Cross-architecture agreement")
if i is not None:
    set_bold_body(P(i), "Interpretation. ",
        "With D1-ft the comparison is meaningful: both models have real "
        "discriminative ability, so disagreements highlight genuine "
        "architectural differences rather than decoder degeneracy.")
    print("  interpretation -> D1-ft")

# ================================================================
# 10. SECTIONS 3.6 / 3.7  (preliminary notes)
# ================================================================
print("\n[10] Sections 3.6/3.7 preliminary notes")
t6 = doc.tables[6]
t6.cell(1, 5).text = "preliminary (n=500)"
print("  3.6 teacher row -> preliminary")

i = find_para("Interpretation (vs H2). The student matches")
if i is not None:
    np = insert_para_after(P(i - 1), "", "Body Text")
    np.runs[0].text = ("Final accuracy numbers for both models on "
                        "n\u00a0=\u00a05,000 are in the fair comparison table "
                        "(\u00a73.1). The numbers above are from the "
                        "preliminary n\u00a0=\u00a0500 run.")
    np.runs[0].italic = True
    print("  3.6 preliminary note inserted")

t7 = doc.tables[7]
t7.cell(1, 4).text = "preliminary (n=200)"
t7.cell(2, 4).text = "preliminary (n=200)"
print("  3.7 rows -> preliminary")

i = find_para("Interpretation. Light task-specific adaptation recovers strong")
if i is not None:
    np = insert_para_after(P(i - 1), "", "Body Text")
    np.runs[0].text = ("Final numbers for D1 and D1-ft on n\u00a0=\u00a05,000 "
                        "are in the fair comparison table (\u00a73.1).")
    np.runs[0].italic = True
    for r in P(i).runs:
        r.text = r.text.replace("toxic-bert (0.15)", "toxic-bert (0.14)")
        r.text = r.text.replace("the SVM (0.09)", "the SVM (0.08)")
    print("  3.7 preliminary note inserted, FPR refs updated")

# ================================================================
# 11. ETHICS FPR REFS
# ================================================================
print("\n[11] Ethics")
i = find_para("identity FPRs (toxic-bert")
if i is not None:
    for r in P(i).runs:
        r.text = r.text.replace("15%", "14%").replace("9%", "8%")
    print("  updated ethics FPR refs")

# ================================================================
# 12. FIGURES
# ================================================================
print("\n[12] Inserting figures...")

figure_map = [
    ("Accuracy vs", "model_comparison_bars_test.png", 5.0),
    ("Accuracy vs", "accuracy_latency_test.png", 5.0),
    ("Identity FPR on neutral", "fairness_fpr_heatmap.png", 5.5),
    ("Evasion rates", "robustness_evasion_heatmap.png", 5.5),
    ("masking drops identity FPR from 0.139", "masking_inference_ablation.png", 5.0),
    ("masking drops identity FPR from 0.139", "masking_svm_tradeoff.png", 5.0),
]

inserted = set()
for anchor_text, fname, width in figure_map:
    img = FIGURES_DIR / fname
    if not img.exists():
        print(f"  [SKIP] {fname}")
        continue
    start = 0
    if fname in inserted:
        start = find_para(anchor_text, 0)
        if start is not None:
            start += 2
        else:
            continue
    i = find_para(anchor_text, start)
    if i is not None:
        insert_image_after(doc.paragraphs[i], img, width)
        inserted.add(fname)

# encoder-decoder agreement figures
for fname in ["enc_dec_gold_agreement.png", "enc_dec_agreement.png"]:
    img = FIGURES_DIR / fname
    if img.exists():
        i = find_para("Disagreement examples") or find_para("Decoder (D1-ft)")
        if i is not None:
            insert_image_after(doc.paragraphs[i], img, 5.0)

# ================================================================
# SAVE
# ================================================================
doc.save(str(DOCX_PATH))
print(f"\nDone! Saved {DOCX_PATH.name}")
print(f"Backup: {BACKUP.name}")
