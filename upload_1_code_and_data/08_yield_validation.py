"""
SCRIPT 8 - Crop validation with official ICRISAT district statistics, Gaya

Checks whether the satellite findings hold for official crop statistics:
  1. Do rice (Kharif) area, production and yield fall in monsoon-drought years, and wheat (Rabi)
     in hot years?
  2. Does a larger irrigated share weaken those losses (anomaly ~ SPEI x share, ~ heat x share)?
  3. Does district VHI (Script 3) track the anomalies (is VHI a valid impact proxy)?

Input (data/): any icrisat*.csv for Gaya (ICRISAT DLD downloads, or the Mendeley copy extracted
to icrisat_gaya_apy_mendeley.csv). Column names are matched loosely; -1 = missing.
Also uses data/gaya_climate_monthly.csv, outputs/annual_drought_indices.csv, outputs/vhi_merged.csv
Run:  python 08_yield_validation.py   ->  outputs_yield/
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT3_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "outputs"
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "outputs_yield"
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
summary = []


def say(t=""):
    print(t)
    summary.append(t)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    se = np.sqrt(np.diag(resid @ resid / dof * np.linalg.inv(X1.T @ X1)))
    return beta, se, 2 * (1 - stats.t.cdf(np.abs(beta / se), dof)), 1 - resid @ resid / np.sum((y - y.mean()) ** 2)


PATTERNS = {"rice_area": r"^RICE AREA", "rice_prod": r"^RICE PRODUCTION", "rice_yield": r"^RICE YIELD",
            "rice_irr": r"^RICE IRRIGATED AREA", "wheat_area": r"^WHEAT AREA", "wheat_prod": r"^WHEAT PRODUCTION",
            "wheat_yield": r"^WHEAT YIELD", "wheat_irr": r"^WHEAT IRRIGATED AREA",
            "gia": r"^GROSS IRRIGATED AREA", "gca": r"^GROSS CROPPED AREA"}
frames = []
for f in sorted(DATA_DIR.glob("icrisat*.csv")):
    d = pd.read_csv(f)
    if "Dist Name" in d.columns:
        d = d[d["Dist Name"].astype(str).str.strip().str.lower() == "gaya"]
    keep = {"Year": "year"}
    for key, pat in PATTERNS.items():
        for c in d.columns:
            if re.search(pat, c.strip().upper()):
                keep[c] = key
    frames.append(d[list(keep)].rename(columns=keep).set_index("year"))
y = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce")
y = y.T.groupby(level=0).first().T          # merge duplicate columns from several files
y = y.where(y >= 0).sort_index()
say("=" * 72)
say("CROP VALIDATION WITH ICRISAT DISTRICT STATISTICS - GAYA")
say("=" * 72)
say(f"Years {y.index.min()}-{y.index.max()}; variables: {', '.join(y.columns)}")

# quality screen: a yield more than 2.5x the crop's median (with no matching rise in production) is a data error
for crop in ["rice", "wheat"]:
    yc = f"{crop}_yield"
    if yc in y.columns:
        bad = y.index[y[yc] > 2.5 * y[yc].median()]
        if len(bad):
            say(f"Quality screen: {crop} years {list(bad)} removed (yield > 2.5 x median, inconsistent with area)")
            y.loc[bad, [c for c in y.columns if c.startswith(crop)]] = np.nan

annual = pd.read_csv(OUT3_DIR / "annual_drought_indices.csv").set_index("year")
cm = pd.read_csv(DATA_DIR / "gaya_climate_monthly.csv")
cm["yr"], cm["mo"] = cm.date.str[:4].astype(int), cm.date.str[5:7].astype(int)
heat = {"rice": cm[cm.mo.isin([8, 9, 10])].groupby("yr").tmax_C.mean(),
        "wheat": cm[cm.mo.isin([2, 3])].groupby("yr").tmax_C.mean().rename(lambda v: v - 1)}
zs = lambda s: (s - s.mean()) / s.std()


def anomaly_pct(s):
    """Percent departure from a robust (Theil-Sen) linear trend."""
    s = s.dropna()
    sl, ic = stats.theilslopes(s.values, s.index.values)[:2]
    tr = ic + sl * s.index.values
    return pd.Series((s.values - tr) / tr * 100, index=s.index)


rows, plots = [], {}
for crop, season in [("rice", "kharif"), ("wheat", "rabi")]:
    metrics = [m for m in ["area", "prod", "yield"] if f"{crop}_{m}" in y.columns and y[f"{crop}_{m}"].notna().sum() > 10]
    if not metrics:
        say(f"\n{crop}: no data (download {crop} area, production and yield from ICRISAT)")
        continue
    share = (y[f"{crop}_irr"] / y[f"{crop}_area"]) if {f"{crop}_irr", f"{crop}_area"} <= set(y.columns) else \
            (y.gia / y.gca if {"gia", "gca"} <= set(y.columns) else None)
    share_name = f"{crop} irrigated share" if f"{crop}_irr" in y.columns else "gross irrigated / gross cropped area"
    vf = OUT3_DIR / "vhi_merged.csv"
    vhi = None
    if vf.exists():
        v = pd.read_csv(vf)
        vhi = v[(v.season == season) & v.group.isin(["Irrigated", "Rainfed"])].groupby("year").VHI.mean()
    for m in metrics:
        d = pd.DataFrame({"anom": anomaly_pct(y[f"{crop}_{m}"])}).join(annual[["SPEI", "drought"]]) \
            .join(zs(heat[crop]).rename("heat_z"))
        if share is not None:
            d["share"] = share
        d = d.dropna(subset=["anom", "SPEI", "heat_z"])
        d["drought"] = d.drought.astype(bool)
        dr, nm = d[d.drought], d[~d.drought]
        u = stats.mannwhitneyu(dr.anom, nm.anom, alternative="less") if len(dr) > 1 else None
        b, se, p, r2 = ols(np.column_stack([d.SPEI, d.heat_z]), d.anom.to_numpy())
        say("")
        say(f"--- {crop.upper()} {m.upper()} ({len(d)} years) ---")
        say(f"Anomaly in drought years {dr.anom.mean():+.1f}% (n={len(dr)}: {list(dr.index)}) vs normal "
            f"{nm.anom.mean():+.1f}%" + (f"  [Mann-Whitney one-sided p = {u.pvalue:.3f}]" if u else ""))
        say(f"Anomaly = {b[0]:+.1f} {b[1]:+.1f}% per SPEI unit (p={p[1]:.3f}) {b[2]:+.1f}% per SD heat (p={p[2]:.3f}), R2={r2:.2f}")
        row = dict(crop=crop, metric=m, n=len(d), drought_anom=dr.anom.mean(), normal_anom=nm.anom.mean(),
                   p_drought=u.pvalue if u else np.nan, b_SPEI=b[1], p_SPEI=p[1], b_heat=b[2], p_heat=p[2], R2=r2)
        if "share" in d.columns and d.share.notna().sum() > 10:
            e = d.dropna(subset=["share"])
            sh = (e.share - e.share.mean()) / e.share.std()            # standardized share
            X = np.column_stack([e.SPEI, e.heat_z, sh, e.SPEI * sh, e.heat_z * sh])
            b2, se2, p2, _ = ols(X, e.anom.to_numpy())
            say(f"With {share_name} (mean {e.share.mean():.2f}): SPEI x share {b2[4]:+.1f} (p={p2[4]:.3f}), "
                f"heat x share {b2[5]:+.1f} (p={p2[5]:.3f})")
            say("   (negative SPEI x share = smaller drought losses when more is irrigated)")
            row.update(b_SPEIxShare=b2[4], p_SPEIxShare=p2[4], b_heatxShare=b2[5], p_heatxShare=p2[5])
        if vhi is not None:
            j = d.join(vhi.rename("VHI"), how="inner").dropna(subset=["VHI"])
            if len(j) > 5:
                r, pr = stats.pearsonr(j.VHI, j.anom)
                say(f"VHI vs {crop} {m} anomaly ({len(j)} years, {j.index.min()}-{j.index.max()}): r = {r:.2f} (p = {pr:.3f})")
                row.update(r_VHI=r, p_VHI=pr, n_VHI=len(j))
        rows.append(row)
        plots[(crop, m)] = d
pd.DataFrame(rows).round(4).to_csv(OUT_DIR / "yield_validation.csv", index=False)

if plots:
    keys = list(plots)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.3), sharey=False)
    for ax, (crop, m) in zip(np.atleast_1d(axes), keys):
        d = plots[(crop, m)]
        xv, lab = (d.SPEI, "Monsoon SPEI-4") if crop == "rice" else (d.heat_z, "Feb-Mar heat (SD)")
        ax.scatter(xv, d.anom, c=np.where(d.drought, "#d95f02", "#1f78b4"), s=18)
        lr = stats.linregress(xv, d.anom)
        xs = np.linspace(xv.min(), xv.max(), 10)
        ax.plot(xs, lr.intercept + lr.slope * xs, color="0.3")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(lab)
        ax.set_ylabel(f"{crop} {m} anomaly (%)")
        ax.set_title(f"r = {lr.rvalue:.2f}, p = {lr.pvalue:.3f}", fontsize=9)
    fig.text(0.01, 0.01, "orange = drought years (SPEI-4 <= -1)", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / "fig13_crop_validation.png")
(OUT_DIR / "yield_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print(f"\nDone. Results in {OUT_DIR}")
