"""
SCRIPT 9 - District-scale check of GRACE with CGWB observation wells, Gaya

Reads the station tables "Depth to water level (m bgl) in various seasons: May Aug Nov Jan" from the
CGWB Ground Water Year Books for Bihar (2018-19, 2019-20, 2020-21) in data/cgwb/, keeps Gaya wells,
and compares the district median water level with GRACE groundwater storage in the same months.

Note: only these three year books list Gaya wells as extractable text; older editions do not.
Run:  python 09_cgwb_wells.py      ->  outputs_novelty/G_cgwb_*.csv, fig14_cgwb_vs_grace.png
"""
import collections
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CG = HERE / "data" / "cgwb"
OUT = HERE / "outputs_novelty"
OUT.mkdir(exist_ok=True)
num = re.compile(r"^\d+(\.\d+)?$")

# book -> (page with Gaya wells (1-based) list, first month's year)
BOOKS = {"GWYB_2018-19_Bihar_hi.pdf": ([50], 2018),
         "GWYB_2019-20_Bihar_hi.pdf": ([51, 52], 2019),
         "GWYB_2020-21_Bihar.pdf": ([50], 2020)}
MONTHS = [("May", 5, 0), ("Aug", 8, 0), ("Nov", 11, 0), ("Jan", 1, 1)]   # (name, month, year offset)

rows = []
for book, (pages, y0) in BOOKS.items():
    r = pypdf.PdfReader(CG / book)
    for pg in pages:
        items = []
        r.pages[pg - 1].extract_text(visitor_text=lambda t, cm, tm, fd, fs: items.append(
            (round(tm[5], 1), round(tm[4], 1), t.strip())) if t.strip() else None)
        lines = collections.defaultdict(list)
        for yy, x, t in items:
            lines[yy].append((x, t))
        gaya = [sorted(v) for v in lines.values() if any(t.lower() == "gaya" for x, t in v)]
        # value columns: cluster the x positions of numeric/'-' cells on this page into 4 groups
        xs = sorted(x for v in gaya for x, t in v if num.match(t) or t == "-")
        if not xs:
            continue
        cuts = np.quantile(xs, [0, 1])
        centers = np.linspace(cuts[0], cuts[1], 4)
        for _ in range(20):                                   # 1-D k-means, k = 4
            lab = np.argmin(np.abs(np.array(xs)[:, None] - centers[None, :]), axis=1)
            centers = np.array([np.mean(np.array(xs)[lab == k]) if np.any(lab == k) else centers[k]
                                for k in range(4)])
        for v in gaya:
            name = " ".join(t for x, t in v if not (num.match(t) or t == "-") and t.lower() != "gaya")
            vals = {}
            for x, t in v:
                if num.match(t):
                    k = int(np.argmin(np.abs(centers - x)))
                    vals[k] = float(t)
            for k, (mname, mo, off) in enumerate(MONTHS):
                if k in vals and vals[k] < 40:
                    rows.append(dict(book=book, well=name.strip(), month=f"{y0 + off}-{mo:02d}",
                                     dtw_m=vals[k]))
wells = pd.DataFrame(rows)
wells.to_csv(OUT / "G_cgwb_gaya_wells.csv", index=False)
med = wells.groupby("month").dtw_m.agg(["median", "count"]).rename(columns={"median": "dtw_median_m"})
print(f"{wells.well.nunique()} Gaya wells, {len(wells)} readings")
print(med)

g = pd.read_csv(OUT / "C_groundwater_monthly.csv", index_col=0, parse_dates=True)
g.index = g.index.strftime("%Y-%m")
comp = med.join(g[["GWSA_grace_mm", "GWSA_clsm_mm"]], how="left")
comp.to_csv(OUT / "G_cgwb_vs_grace.csv")
res = []
for col in ["GWSA_grace_mm", "GWSA_clsm_mm"]:
    c = comp.dropna(subset=[col])
    if len(c) > 3:
        rho, p = stats.spearmanr(-c.dtw_median_m, c[col])          # shallower water = more storage
        res.append(f"{col}: Spearman rho = {rho:.2f} (p = {p:.3f}, n = {len(c)} months)")
seas = wells.assign(mo=wells.month.str[5:7]).pivot_table(index=["book", "well"], columns="mo", values="dtw_m")
if {"05", "11"} <= set(seas.columns):
    rise = (seas["05"] - seas["11"]).dropna()
    res.append(f"Median May-to-November water-level rise: {rise.median():.2f} m (n = {len(rise)} well-years); "
               f"with a specific yield of 0.10-0.15 this is about {rise.median() * 100:.0f}-{rise.median() * 150:.0f} mm")
for s in res:
    print(s)
(OUT / "G_cgwb_summary.txt").write_text("\n".join([f"{wells.well.nunique()} Gaya wells, {len(wells)} readings",
                                                    med.to_string()] + res), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax2 = ax.twinx()
x = pd.to_datetime(comp.index + "-01")
ax.plot(x, comp.dtw_median_m, "o-", color="#d95f02", label="CGWB wells, median depth (m)")
ax.invert_yaxis()
ax2.plot(x, comp.GWSA_grace_mm, "s--", color="#1f78b4", label="GRACE storage anomaly (mm)")
ax.set_ylabel("Depth to water (m bgl)")
ax2.set_ylabel("GRACE GWSA (mm)")
ax.set_title("Gaya wells vs GRACE, 2018-2021")
fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "fig14_cgwb_vs_grace.png", dpi=300)
