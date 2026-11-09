"""
SCRIPT 10 - South Bihar regional replication (uses outputs of Script 2b)

Splits region_vhi_by_group.csv and region_climate_monthly.csv by district, runs Script 3
(drought buffering) and Script 7 (paired heat/SPEI test, robustness) for each district,
and collects one summary table.

Run:  python 10_regional_analysis.py   ->  outputs_region/regional_summary.csv, fig15_regional_buffering.png
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs_region"
OUT.mkdir(exist_ok=True)
CANAL = {"Rohtas", "Bhojpur"}          # Sone canal command (contrast districts)

vhi = pd.read_csv(DATA / "region_vhi_by_group.csv")
clim = pd.read_csv(DATA / "region_climate_monthly.csv")
rows = []
for dist in sorted(vhi.district.unique()):
    d_in, d3, d7 = OUT / dist / "data", OUT / dist / "out3", OUT / dist / "out7"
    d_in.mkdir(parents=True, exist_ok=True)
    vhi[vhi.district == dist].drop(columns="district").to_csv(d_in / "gaya_vhi_by_group.csv", index=False)
    clim[clim.district == dist].drop(columns="district")[["date", "pr_mm", "tmax_C", "tmin_C"]] \
        .sort_values("date").to_csv(d_in / "gaya_climate_monthly.csv", index=False)
    for script, args in [("03_drought_buffering_analysis.py", [d_in, d3]),
                         ("07_novelty_analysis.py", [d_in, d3, d7])]:
        subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    buf = pd.read_csv(d3 / "buffering_results.csv")
    reg = pd.read_csv(d7 / "B_compound_regression.csv")
    for season in ["kharif", "rabi"]:
        b = buf[buf.season == season].set_index("group")
        gp = reg[(reg.season == season) & reg.group.str.startswith("gap")]
        if b.empty:
            continue
        drop = b.VHI_normal - b.VHI_drought
        rows.append(dict(district=dist, canal_command=dist in CANAL, season=season,
                         n_years=int(b.n_years.max()),
                         drop_irrigated=drop.get("Irrigated", np.nan), drop_rainfed=drop.get("Rainfed", np.nan),
                         buffering_points=drop.get("Rainfed", np.nan) - drop.get("Irrigated", np.nan),
                         gap_vs_SPEI=gp.b_SPEI.iloc[0] if len(gp) else np.nan,
                         p_SPEI=gp.p_SPEI.iloc[0] if len(gp) else np.nan,
                         gap_vs_heat=gp.b_heat.iloc[0] if len(gp) else np.nan,
                         p_heat=gp.p_heat.iloc[0] if len(gp) else np.nan))
    print(f"{dist}: done")

s = pd.DataFrame(rows)
s.round(3).to_csv(OUT / "regional_summary.csv", index=False)
print(s.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, season in zip(axes, ["kharif", "rabi"]):
    d = s[s.season == season].sort_values("buffering_points")
    col = ["#1b9e77" if c else ("#1f78b4" if n == "Gaya" else "#999999") for n, c in zip(d.district, d.canal_command)]
    ax.barh(d.district, d.buffering_points, color=col)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"{season.capitalize()}: rainfed drop minus irrigated drop")
    ax.set_xlabel("VHI points (positive = irrigation buffers drought)")
fig.text(0.01, 0.01, "green = Sone canal command; blue = Gaya", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "fig15_regional_buffering.png", dpi=300)
