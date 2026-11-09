"""
SCRIPT 11 - Agreement between the two interpreters (inter-rater reliability)

Input: data/second_interpreter_labels.txt  -> the COPY BOX line from Script 1e, e.g.
       var RESUME = [[84.731622,24.507512,0],[...]];
       data/gaya_validation_points_2023.csv (first interpreter, Script 1b)
Run:   python 11_inter_rater.py   ->  outputs_novelty/H_inter_rater.txt

Repeat (intra-rater) check by the FIRST interpreter, blind, days later:
       save the COPY BOX line as data/repeat_labels.txt
Run:   python 11_inter_rater.py repeat   ->  outputs_novelty/H_intra_rater.txt
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
REPEAT = len(sys.argv) > 1 and sys.argv[1] == "repeat"
WHO = ["first interpreter", "repeat labelling"] if REPEAT else ["first interpreter", "second interpreter"]
txt = (HERE / "data" / ("repeat_labels.txt" if REPEAT else "second_interpreter_labels.txt")).read_text(encoding="utf-8")
second = json.loads(re.search(r"\[\[.*\]\]", txt, re.S).group(0))
s2 = pd.DataFrame(second, columns=["lon", "lat", "ref2"])
v = pd.read_csv(HERE / "data" / "gaya_validation_points_2023.csv")
v["lon"] = v[".geo"].apply(lambda g: round(json.loads(g)["coordinates"][0], 5))
v["lat"] = v[".geo"].apply(lambda g: round(json.loads(g)["coordinates"][1], 5))
s2["lon"], s2["lat"] = s2.lon.round(5), s2.lat.round(5)
m = s2.merge(v[["lon", "lat", "ref", "map"]], on=["lon", "lat"])
m = m[(m.ref >= 0) & (m.ref2 >= 0)]
names = ["Rabi-cropped", "Rainfed", "Fallow"]
lines = [("REPEAT (INTRA-RATER) CHECK, same interpreter, blind" if REPEAT else "INTER-RATER CHECK, two interpreters"),
         f"Points with a class in both labellings: {len(m)}",
         f"Three classes: agreement {np.mean(m.ref == m.ref2) * 100:.0f}%, Cohen's kappa {cohen_kappa_score(m.ref, m.ref2):.2f}",
         f"Rabi-cropped vs not: agreement {np.mean((m.ref == 0) == (m.ref2 == 0)) * 100:.0f}%, "
         f"kappa {cohen_kappa_score(m.ref == 0, m.ref2 == 0):.2f}",
         f"Confusion (rows = {WHO[0]}, columns = {WHO[1]}):",
         pd.DataFrame(confusion_matrix(m.ref, m.ref2, labels=[0, 1, 2]), index=names, columns=names).to_string(),
         f"Map agreement with the {WHO[1]} (three classes): {np.mean(m['map'] == m.ref2) * 100:.0f}%"]
print("\n".join(lines))
out = "H_intra_rater.txt" if REPEAT else "H_inter_rater.txt"
(HERE / "outputs_novelty" / out).write_text("\n".join(lines), encoding="utf-8")
