"""
SCRIPT 14b - Figure 1: study area (India inset, Bihar inset, Gaya with the 2023-24 RF map)

Inputs (from Script 14, unzipped into data/figure1/):
    fig1_rabi_map_30m.tif, fig1_gaya_boundary.shp, fig1_region_districts.shp,
    fig1_bihar_districts.shp, fig1_india_states.geojson
Run:    python 14b_figure1_study_area.py   ->  outputs/Figure1_study_area.png (400 dpi) and .pdf
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
F = HERE / "data" / "figure1"
OUT = HERE / "outputs"

gaya = gpd.read_file(F / "fig1_gaya_boundary.shp")
region = gpd.read_file(F / "fig1_region_districts.shp")
bihar = gpd.read_file(F / "fig1_bihar_districts.shp")

with rasterio.open(F / "fig1_rabi_map_30m.tif") as src:
    r = src.read(1, masked=True)
    b = src.bounds
    ext = (b.left, b.right, b.bottom, b.top)
r = np.ma.masked_where(np.ma.getmaskarray(r) | (r == 0) | (r > 3), r) - 1   # file: 1-3 = classes, 0 = no data

COLORS = ["#1f78b4", "#e6ab02", "#d9d9d9"]
LABELS = ["Rabi-cropped (irrigated)", "Rainfed (Kharif only)", "Fallow / other"]

fig = plt.figure(figsize=(7.2, 6.0))
ax = fig.add_axes([0.37, 0.08, 0.61, 0.86])
ax.imshow(r, extent=ext, cmap=ListedColormap(COLORS), vmin=0, vmax=2, interpolation="nearest")
gaya.boundary.plot(ax=ax, color="black", lw=0.9)
x0, y0, x1, y1 = gaya.total_bounds
ax.set_xlim(x0 - 0.05, x1 + 0.05)
ax.set_ylim(y0 - 0.05, y1 + 0.05)
ax.set_xlabel("Longitude (°E)", fontsize=8)
ax.set_ylabel("Latitude (°N)", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title("(b) Gaya district: Random Forest map, Rabi 2023–24", fontsize=9)
ax.legend(handles=[Patch(fc=c, ec="none", label=l) for c, l in zip(COLORS, LABELS)],
          loc="upper left", fontsize=7, frameon=True)
# north arrow and scale bar (1° longitude at ~24.7°N ~ 101 km)
ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.86), xycoords="axes fraction", ha="center",
            fontsize=9, weight="bold", arrowprops=dict(arrowstyle="-|>", color="black"))
km20 = 20 / (111.32 * np.cos(np.radians((y0 + y1) / 2)))
sx, sy = x1 - km20 - 0.05, y0 - 0.02
ax.plot([sx, sx + km20], [sy, sy], color="black", lw=2)
ax.text(sx + km20 / 2, sy + 0.015, "20 km", ha="center", fontsize=7)

# inset: Bihar with the study region and Gaya (no India inset: the GAUL national
# boundary differs from the Survey of India boundary)
a2 = fig.add_axes([0.01, 0.30, 0.30, 0.45])
bihar.plot(ax=a2, color="#f0f0f0", ec="#9e9e9e", lw=0.3)
region.plot(ax=a2, color="#fdd0a2", ec="#7f7f7f", lw=0.4)
gaya.plot(ax=a2, color="#e6550d", ec="black", lw=0.5)
a2.set_title("(a) Bihar: Gaya (dark) and\nsix neighbouring districts", fontsize=7.5)
a2.set_axis_off()

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "Figure1_study_area.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "Figure1_study_area.pdf", bbox_inches="tight", pad_inches=0.05)
print("saved", OUT / "Figure1_study_area.png")
