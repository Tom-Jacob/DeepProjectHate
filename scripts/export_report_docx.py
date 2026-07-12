"""Export report/Words_That_Harm_Report.docx from report_draft.md."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Cm, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "report" / "report_draft.md"
FIGS = ROOT / "report" / "figures"
OUT = ROOT / "report" / "Words_That_Harm_Report.docx"

FIG_MAP = {
    "model_comparison_bars_test.png": "Figure: model comparison bars (test)",
    "accuracy_latency_test.png": "Figure: accuracy vs latency (test)",
    "fairness_fpr_heatmap.png": "Figure: per-identity FPR heatmap",
    "robustness_evasion_heatmap.png": "Figure: robustness / evasion heatmap",
    "masking_inference_ablation.png": "Figure: inference masking ablation",
    "masking_svm_tradeoff.png": "Figure: SVM masking fairness/recall trade-off",
    "enc_dec_agreement.png": "Figure: encoder–decoder agreement",
    "enc_dec_gold_agreement.png": "Figure: encoder/decoder vs gold agreement",
}


def style_normal(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE


def add_heading(doc: Document, text: str, level: int) -> None:
    doc.add_heading(text.lstrip("# ").strip(), level=min(level, 3))


def add_para(doc: Document, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold


def main() -> None:
    lines = DRAFT.read_text(encoding="utf-8").splitlines()
    doc = Document()
    style_normal(doc)
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        raw = line.rstrip()

        if raw.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = doc.add_paragraph()
            run = p.add_run(raw)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            i += 1
            continue

        if raw.startswith("> "):
            add_para(doc, raw[2:], bold=False)
            i += 1
            continue

        if raw.startswith("# "):
            add_heading(doc, raw, 1)
            i += 1
            continue
        if raw.startswith("## "):
            add_heading(doc, raw, 2)
            i += 1
            continue
        if raw.startswith("### "):
            add_heading(doc, raw, 3)
            i += 1
            continue

        # Inline figure markers like *Figures: `foo.png`.*
        if "figures/" in raw.lower() or ".png" in raw:
            for name, caption in FIG_MAP.items():
                if name in raw:
                    img = FIGS / name
                    if img.exists():
                        doc.add_picture(str(img), width=Inches(5.8))
                        cap = doc.add_paragraph()
                        r = cap.add_run(caption)
                        r.italic = True
                        r.font.size = Pt(10)
            # still add the text line
            if raw:
                add_para(doc, raw.replace("**", "").replace("*", "").replace("`", ""))
            i += 1
            continue

        if raw.startswith("|") and "|" in raw[1:]:
            # Collect markdown table
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in row):
                    rows.append(row)
                i += 1
            if rows:
                table = doc.add_table(rows=len(rows), cols=len(rows[0]))
                table.style = "Table Grid"
                for r_idx, row in enumerate(rows):
                    for c_idx, cell in enumerate(row):
                        table.rows[r_idx].cells[c_idx].text = cell.replace("**", "")
            continue

        if raw.startswith("- ") or raw.startswith("* "):
            doc.add_paragraph(raw[2:].replace("**", "").replace("`", ""), style="List Bullet")
            i += 1
            continue

        if not raw:
            i += 1
            continue

        # Plain paragraph — strip light markdown
        text = raw.replace("**", "").replace("`", "")
        add_para(doc, text)
        i += 1

    # Append any remaining figures not already referenced
    doc.add_heading("Appendix — report figures", level=2)
    for name, caption in FIG_MAP.items():
        img = FIGS / name
        if img.exists():
            doc.add_picture(str(img), width=Inches(5.8))
            cap = doc.add_paragraph()
            r = cap.add_run(caption)
            r.italic = True

    try:
        doc.save(str(OUT))
        print("saved", OUT)
    except PermissionError:
        alt = OUT.with_name("Words_That_Harm_Report_updated.docx")
        doc.save(str(alt))
        print("original locked; saved", alt)


if __name__ == "__main__":
    main()
