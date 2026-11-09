"""
SCRIPT 17 - Cover letter as Word: submission/cover_letter_EMA.md -> submission/Cover_Letter_EMA.docx
Handles the markdown used in the letter: **bold**, *italic*, "- " bullets and "1. " numbered items.

Run:    python 17_cover_letter_docx.py
"""
import re
from pathlib import Path

import docx
from docx.shared import Pt

SUB = Path(__file__).resolve().parent / "submission"
d = docx.Document()
st = d.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(11)
st.paragraph_format.space_after = Pt(6)


def add_runs(par, text):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and len(part) > 1:
            par.add_run(part[1:-1]).italic = True
        elif part:
            par.add_run(part)


lines = (SUB / "cover_letter_EMA.md").read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    line = lines[i].rstrip()
    if not line:
        i += 1
        continue
    if line.startswith("- "):
        add_runs(d.add_paragraph(style="List Bullet"), line[2:])
    elif re.match(r"\d+\. ", line):
        # literal numbers: Word's "List Number" style would continue counting across separate lists
        p = d.add_paragraph()
        p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Pt(18), Pt(-18)
        num, rest = line.split(" ", 1)
        p.add_run(num + "\t")
        add_runs(p, rest)
    else:
        # consecutive non-blank, non-list lines form one block (address, signature) joined by line breaks
        block = [line]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"(- |\d+\. )", lines[i + 1]) \
                and len(line) < 60 and len(lines[i + 1]) < 60:
            i += 1
            block.append(lines[i].rstrip())
        p = d.add_paragraph()
        for k, b in enumerate(block):
            if k:
                p.add_run().add_break()
            add_runs(p, b)
    i += 1

out = SUB / "Cover_Letter_EMA.docx"
d.save(out)
print("saved", out)
