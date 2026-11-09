"""
SCRIPT 15 - Figures 3-5 for the journal (600 dpi, no titles inside the figure)

  Fig 3  Rabi-cropped area by year (Script 1b) and official irrigated wheat area (ICRISAT, 2015-2020)
  Fig 4  Annual groundwater storage anomaly, GRACE minus GLDAS (Script 7), drought years highlighted, Sen trend
  Fig 5  Permutation importance of the drivers of Kharif drought sensitivity (Script 7)

Run:    python 15_figures_3_5.py   ->  outputs/Figure3_rabi_area.png, Figure4_groundwater.png, Figure5_drivers.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import theilslopes

HERE = Path(__file__).resolve().parent
DATA, NOV, OUT = HERE / "data", HERE / "outputs_novelty", HERE / "outputs"
GREY, BLUE, INK = "#c3c3b6", "#2a78d6", "#3c3c3c"
DROUGHT_YEARS = [2009, 2010, 2022, 2023]   # SPEI-4 (Sep) <= -1.0 within the GRACE record
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


# Fig 3 - Rabi-cropped area by agricultural year
a = pd.read_csv(DATA / "gaya_irrigated_area_by_year.csv")
a["label"] = a.year.astype(str) + "–" + (a.year + 1).astype(str).str[2:]
a["kha"] = a.ha_class0 / 1000
w = pd.read_csv(DATA / "icrisat_gaya_wheat_irrigated.csv")
wheat = w[w.Year.between(2015, 2020)]["WHEAT IRRIGATED AREA (1000 ha)"].astype(float).mean()
fig, ax = plt.subplots(figsize=(6.85, 3.0))
bars = ax.bar(a.label, a.kha, color=[BLUE if y == 2023 else GREY for y in a.year], width=0.62)
for b, v in zip(bars, a.kha):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
ax.axhline(wheat, color=INK, ls="--", lw=0.8)
ax.text(len(a) - 0.5, wheat + 6, f"Official irrigated wheat, 2015–2020 mean: {wheat:.1f}", ha="right", fontsize=7,
        bbox=dict(fc="white", ec="none", pad=1.5))
ax.set_ylabel("Rabi-cropped area (thousand ha)")
ax.set_xlabel("Agricultural year (2023–24 = validated map)")
ax.set_ylim(0, 300)
save(fig, "Figure3_rabi_area")

# Fig 4 - annual groundwater storage anomaly
g = pd.read_csv(NOV / "C_groundwater_monthly.csv")
g["year"] = pd.to_datetime(g.iloc[:, 0]).dt.year
ann = g.groupby("year").GWSA_grace_mm.agg(["mean", "count"])
ann = ann[ann["count"] >= 8]["mean"]
slope, icpt, _, _ = theilslopes(ann.values, ann.index.values)
fig, ax = plt.subplots(figsize=(6.85, 3.0))
ax.bar(ann.index, ann.values, color=[BLUE if y in DROUGHT_YEARS else GREY for y in ann.index], width=0.75)
ax.plot(ann.index, icpt + slope * ann.index, color=INK, ls="--", lw=0.9)
ax.text(ann.index.max(), icpt + slope * ann.index.max() - 12, f"Sen trend {slope * 10:+.0f} mm per decade",
        ha="right", va="top", fontsize=7)
ax.axhline(0, color=INK, lw=0.5)
ax.set_ylabel("Groundwater storage anomaly (mm)\nrelative to 2004–2009")
ax.set_xlabel("Year (blue = drought monsoon; 2016–2018 incomplete GRACE record)")
ax.set_xticks(range(2002, 2025, 2))
save(fig, "Figure4_groundwater")

# Fig 5 - driver importance
names = {"elev_m": "Elevation", "rain_cv": "Monsoon rainfall variability (CV)", "mar_tmax_C": "March maximum temperature",
         "clay_pct": "Soil clay content", "slope_deg": "Terrain slope", "wc_irr_frac": "WorldCereal irrigated fraction",
         "dist_river_m": "Distance to river", "irr_index": "Irrigation index (Sentinel-2)"}
d = pd.read_csv(NOV / "D_driver_importance.csv").sort_values("importance")
fig, ax = plt.subplots(figsize=(6.85, 3.0))
irr = d.driver.isin(["wc_irr_frac", "irr_index"])
ax.barh(d.driver.map(names), d.importance, xerr=d.sd, color=[BLUE if i else GREY for i in irr],
        error_kw=dict(ecolor=INK, lw=0.7, capsize=0), height=0.6)
ax.set_xlabel("Permutation importance on held-out spatial blocks (drop in R², mean ± SD of 5 folds)\n"
              "blue = irrigation variables")
ax.set_xlim(0, None)
save(fig, "Figure5_drivers")
