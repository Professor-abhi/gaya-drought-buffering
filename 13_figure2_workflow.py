"""
SCRIPT 13 - Figure 2: methodological workflow (replaces the draw.io placeholder)

Run:    python 13_figure2_workflow.py   ->  outputs/Figure2_workflow.png (400 dpi) and .pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

DATA_C, STEP_C, ROB_C, RES_C = "#dbe9f6", "#fdf0d5", "#e5f2e0", "#f3dede"
EDGE = "#3c3c3c"

fig, ax = plt.subplots(figsize=(7.2, 8.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 120)
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec=EDGE, lw=0.8))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, weight="bold")
    ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=6.4, linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.9, shrinkA=0, shrinkB=0))


# 1. input data
ax.text(50, 119, "Input data", ha="center", va="top", fontsize=8.5, weight="bold")
data = [("Sentinel-2 / Sentinel-1", "2023–24 Rabi season\n10 m"),
        ("MODIS NDVI and LST", "MOD13Q1, MOD11A2\n2000–2025, 250 m / 1 km"),
        ("CHIRPS and ERA5-Land", "rainfall, temperature\n1981–2025"),
        ("GRACE/GRACE-FO, GLDAS", "JPL mascons, Noah 2.1,\nCLSM 2.2, 2002–2024")]
for i, (t, b) in enumerate(data):
    box(1 + i * 25, 101, 22.5, 13, t, b, DATA_C)
data2 = [("WorldCereal / WorldCover", "ESA 2021 irrigation, cropland"),
         ("CGWB observation wells", "23 Gaya wells, 2018–21"),
         ("ICRISAT DLD and DES", "district crop statistics"),
         ("Six neighbouring districts", "South Bihar replication")]
for i, (t, b) in enumerate(data2):
    box(1 + i * 25, 89, 22.5, 9, t, b, DATA_C)

# 2. main steps
box(1, 64, 31, 20, "1. Irrigation mapping",
    "Automatic crop-calendar labels\nRandom Forest (100 trees)\nRabi-cropped vs rainfed, 10 m\nGaya 2023–24", STEP_C)
box(34.5, 64, 31, 20, "2. Accuracy and agreement",
    "163 blind reference points\nOlofsson error-adjusted area\nWorldCereal and ICRISAT\nFleiss' κ; blind repeat labelling", STEP_C)
box(68, 64, 31, 20, "3. Drought and heat indices",
    "SPI-4, SPEI-4 (Sept.)\nVCI, TCI, VHI per pixel\nKharif and Rabi seasons\nFeb–Mar Tmax (heat)", STEP_C)
for x in (16.5, 50, 83.5):
    arrow(x, 88.4, x, 84.6)

box(1, 37, 31, 21, "4. Buffering tests",
    "Detrended drop in drought years\nirrigated vs rainfed pixels\nPaired year-matched gap\n= a + c1·SPEI + c2·heat\nBootstrap CIs, Holm correction", STEP_C)
box(34.5, 37, 31, 21, "5. Groundwater",
    "GRACE TWS − GLDAS soil water\nmonsoon recharge, drawdown\nMann–Kendall / Sen trend\nCGWB well comparison", STEP_C)
box(68, 37, 31, 21, "6. Drivers",
    "Per-pixel drought sensitivity\nRandom Forest with\nspatial-block cross-validation\npermutation importance", STEP_C)
arrow(16.5, 63.4, 16.5, 58.6)
arrow(83.5, 63.4, 83.5, 58.6)
arrow(50, 63.4, 50, 58.6)

# 3. robustness
box(1, 18, 98, 13, "Robustness checks",
    "Persistent groups · rainfed Rabi crops · VCI and NDVI without LST · matching on terrain and soil\n"
    "Equivalence test · each drought year · Sentinel-2 20 m check (2022–23) · DES crop yields · six districts",
    ROB_C)
for x in (16.5, 50, 83.5):
    arrow(x, 36.4, x, 31.6)

# 4. findings
box(1, 1, 31, 12, "No average drought buffering", "Kharif VHI drop irrigated ≈ rainfed;\n≥ 25% ruled out; modest in 2022–23", RES_C)
box(34.5, 1, 31, 12, "Heat buffering (crop condition)", "Irrigation advantage grows with\nFeb–Mar heat (p ≈ 0.002); not in yields", RES_C)
box(68, 1, 31, 12, "Regional groundwater decline", "Storage falls ≈ 50 mm per decade\nas irrigated wheat expands", RES_C)
for x in (16.5, 50, 83.5):
    arrow(x, 17.4, x, 13.6)

fig.tight_layout(pad=0.3)
fig.savefig(OUT / "Figure2_workflow.png", dpi=400)
fig.savefig(OUT / "Figure2_workflow.pdf")
print("saved", OUT / "Figure2_workflow.png")
