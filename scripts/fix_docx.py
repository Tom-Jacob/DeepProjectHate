"""Fix heading and image placement issues in WordsThatHarm.docx."""
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
DOCX_PATH = REPORT_DIR / "WordsThatHarm.docx"
FIGURES_DIR = REPORT_DIR / "figures"

doc = Document(str(DOCX_PATH))
paras = doc.paragraphs


def find_para(substr, start=0):
    for i in range(start, len(paras)):
        if substr in paras[i].text:
            return i
    return None


def has_image(para):
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    ns2 = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for r in para.runs:
        if (r._element.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing")
            or r._element.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pict")):
            return True
        drawings = r._element.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing")
        if drawings:
            return True
    blip_ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for child in para._element.iter():
        if 'drawing' in child.tag or 'pict' in child.tag:
            return True
    return False


# Fix 1: Restore 3.1 heading
# The heading at para 49 says "Accuracy vs. efficiency (fair comparison..."
# It should say "3.1\tAccuracy vs. Efficiency"
print("[Fix 1] Restoring 3.1 heading")
h_idx = find_para("Accuracy vs")
if h_idx is not None and paras[h_idx].style.name == "Heading 2":
    for r in paras[h_idx].runs:
        r.text = ""
    if paras[h_idx].runs:
        paras[h_idx].runs[0].text = "3.1\tAccuracy vs.\u00a0Efficiency"
    print(f"  Para {h_idx}: restored heading text")

# Fix 2: Move images to after interpretation (end of 3.1 section)
# Currently images are at paras 50, 51 (right after heading)
# They should be after interpretation (currently at para 55)
print("\n[Fix 2] Moving 3.1 images to after interpretation")
# Find the image paragraphs
img_paras = []
for i in range(h_idx + 1, min(h_idx + 5, len(paras))):
    if has_image(paras[i]):
        img_paras.append(i)
        print(f"  Found image at para {i}")

if img_paras:
    interp_idx = find_para("Interpretation. The linear model is close")
    if interp_idx is not None:
        print(f"  Moving images to after para {interp_idx}")
        for img_i in reversed(img_paras):
            elem = paras[img_i]._element
            parent = elem.getparent()
            parent.remove(elem)
            paras[interp_idx]._element.addnext(elem)
        print("  Images moved")

# Fix 3: Also fix the duplicate caption  
# There's now both "Accuracy vs. efficiency (fair comparison...)" as the old
# Table Caption and "Accuracy vs. efficiency (test set)." 
# Update the old one
paras = doc.paragraphs  # refresh
cap_idx = find_para("Accuracy vs.\u00a0efficiency (test set)")
if cap_idx is not None:
    for r in paras[cap_idx].runs:
        r.text = ""
    paras[cap_idx].runs[0].text = (
        "Accuracy vs.\u00a0efficiency (fair comparison, n\u00a0=\u00a05,000). "
        "Placeholder\u2014update after Colab run.")
    print(f"\n[Fix 3] Updated table caption at para {cap_idx}")

# Fix 4: Move fairness image to after table+interpretation (not between caption and interp)
print("\n[Fix 4] Checking fairness image placement")
paras = doc.paragraphs
fair_img_idx = None
for i in range(len(paras)):
    if has_image(paras[i]):
        if find_para("Identity FPR", max(0, i-3)) is not None and i < find_para("3.3", 0):
            fair_img_idx = i
            break

if fair_img_idx is not None:
    fair_interp_idx = find_para("Interpretation. Control FPR is")
    if fair_interp_idx is not None and fair_img_idx < fair_interp_idx:
        print(f"  Moving fairness image from {fair_img_idx} to after {fair_interp_idx}")
        elem = paras[fair_img_idx]._element
        parent = elem.getparent()
        parent.remove(elem)
        paras[fair_interp_idx]._element.addnext(elem)
        print("  Fairness image moved")

# Fix 5: Move robustness image similarly
print("\n[Fix 5] Checking robustness image placement")
paras = doc.paragraphs
for i in range(len(paras)):
    if has_image(paras[i]):
        rob_heading = find_para("3.3")
        rob_interp = find_para("Interpretation. The encoder is far the most robust")
        if rob_heading and rob_interp and rob_heading < i < rob_interp:
            print(f"  Moving robustness image from {i} to after {rob_interp}")
            elem = paras[i]._element
            parent = elem.getparent()
            parent.remove(elem)
            paras[rob_interp]._element.addnext(elem)
            print("  Robustness image moved")
            break

# Fix 6: Move masking images to after interpretation
print("\n[Fix 6] Checking masking image placement")
paras = doc.paragraphs
mask_interp = find_para("The bias is a single-token shortcut")
if mask_interp is not None:
    mask_images = []
    for i in range(len(paras)):
        if has_image(paras[i]):
            mask_heading = find_para("3.4")
            next_heading = find_para("3.5")
            if mask_heading and next_heading and mask_heading < i < next_heading:
                mask_images.append(i)
    if mask_images:
        print(f"  Found {len(mask_images)} masking image(s)")
        for mi in reversed(mask_images):
            if mi < mask_interp:
                elem = paras[mi]._element
                parent = elem.getparent()
                parent.remove(elem)
                paras[mask_interp]._element.addnext(elem)
                print(f"  Moved masking image from {mi} to after interpretation")
                paras = doc.paragraphs

# Fix 7: Move agreement images to after interpretation
print("\n[Fix 7] Checking agreement image placement")
paras = doc.paragraphs
agree_interp = find_para("With D1-ft the comparison is meaningful")
if agree_interp is not None:
    agree_images = []
    for i in range(len(paras)):
        if has_image(paras[i]):
            agree_heading = find_para("3.5")
            next_heading = find_para("3.6")
            if agree_heading and next_heading and agree_heading < i < next_heading:
                agree_images.append(i)
    if agree_images:
        print(f"  Found {len(agree_images)} agreement image(s)")
        for ai in reversed(agree_images):
            if ai < agree_interp:
                elem = paras[ai]._element
                parent = elem.getparent()
                parent.remove(elem)
                paras[agree_interp]._element.addnext(elem)
                print(f"  Moved agreement image from {ai} to after interpretation")
                paras = doc.paragraphs

doc.save(str(DOCX_PATH))
print(f"\nSaved fixes to {DOCX_PATH.name}")
