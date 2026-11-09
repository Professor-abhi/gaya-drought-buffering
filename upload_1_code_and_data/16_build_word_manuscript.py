"""
SCRIPT 16 - Build the Word manuscript for Environmental Monitoring and Assessment

Takes the .docx exported from the manuscript doc and:
  - replaces the date/author line with a title-page block (placeholders to fill in)
  - inserts Figs 1-2 above their captions and swaps the low-resolution chart images (Figs 3-5)
    for the 600 dpi versions, removing the chart data-source lines
  - sets Times New Roman, continuous line numbers and page numbers (journal requirements)
  - writes Fig1.tif ... Fig5.tif for separate upload

Run:    python 16_build_word_manuscript.py   ->  submission/Manuscript_EMA.docx, submission/figures/
"""
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt
from PIL import Image

HERE = Path(__file__).resolve().parent
SUB, OUT = HERE / "submission", HERE / "outputs"
FIGS = {1: "Figure1_study_area.png", 2: "Figure2_workflow.png", 3: "Figure3_rabi_area.png",
        4: "Figure4_groundwater.png", 5: "Figure5_drivers.png"}
WIDTH = {1: 174, 2: 140, 3: 174, 4: 174, 5: 174}   # mm; 174 mm = full page width (Springer)

d = docx.Document(SUB / "Manuscript_export.docx")


def picture_paragraph_before(par, fig):
    new = par.insert_paragraph_before()
    new.alignment = WD_ALIGN_PARAGRAPH.CENTER
    new.add_run().add_picture(str(OUT / FIGS[fig]), width=Mm(WIDTH[fig]))
    return new


def delete(par):
    par._element.getparent().remove(par._element)


# 1. title page block in place of the "date · @author" line
title_info = ["[Author full name]¹*",
              "¹ [Department, Institution, City, State, India]",
              "* Corresponding author: [e-mail] · ORCID: [0000-0000-0000-0000]"]
p1 = d.paragraphs[1]
for line in title_info:
    p1.insert_paragraph_before(line)
delete(p1)

# 2. figures: insert above each caption; drop old chart images and their data-source lines
for par in list(d.paragraphs):
    t = par.text.strip()
    if par._p.xpath(".//w:drawing") and not t:
        delete(par)                                   # old 1344-px chart image
    elif t.startswith(("gaya_irrigated_area_by_year.csv", "gaya_grace_monthly.csv", "D_driver_importance.csv")):
        delete(par)                                   # chart data-source line
for par in list(d.paragraphs):
    t = par.text.strip()
    for n in FIGS:
        if t.startswith(f"Fig. {n} ") or t.startswith(f"Fig. {n} "):
            picture_paragraph_before(par, n)

# table captions stay on the same page as their table
for par in d.paragraphs:
    if par.text.startswith("Table ") and par.text[6:8].strip(" ").isdigit():
        par.paragraph_format.keep_with_next = True

# 3. typography, line numbers, page numbers
for name in ["Normal", "Heading 1", "Heading 2", "Heading 3"]:
    st = d.styles[name]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
d.styles["Normal"].font.size = Pt(11)
for sec in d.sections:
    sp = sec._sectPr
    for old in sp.findall(qn("w:lnNumType")):
        sp.remove(old)
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    sp.append(ln)
    fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for tag, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)

out = SUB / "Manuscript_EMA.docx"
d.save(out)
print("saved", out)

# 4. separate figure files (TIFF, LZW) for upload
(SUB / "figures").mkdir(exist_ok=True)
for n, f in FIGS.items():
    im = Image.open(OUT / f).convert("RGB")
    dpi = 600 if n >= 3 else 400
    im.save(SUB / "figures" / f"Fig{n}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
print("saved", SUB / "figures")
