"""
SCRIPT 3 - Drought buffering by irrigation, Gaya district (Bihar)

Inputs (CSV files exported to Google Drive by Scripts 1 and 2; put them in ./data):
    gaya_climate_monthly.csv         (Script 2)  date, pr_mm, tmax_C, tmin_C
    gaya_vhi_by_group.csv            (Script 2)  year, season, grp, NDVI, LST, VCI, TCI, VHI
    gaya_pixel_sensitivity.csv       (Script 2)  slope, r, irr_index, grp, .geo
    gaya_irrigated_area_by_year.csv  (Script 1, optional)

Run:
    pip install pandas numpy scipy matplotlib
    python 03_drought_buffering_analysis.py            # uses ./data and ./outputs
    python 03_drought_buffering_analysis.py DATA OUT   # custom folders

Outputs (./outputs): tables (CSV), figures (PNG, 300 dpi), results_summary.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "outputs"
LAT_DEG = 24.75            # Gaya district centre
DROUGHT_T = -1.0           # SPEI <= -1 -> drought year (moderate or worse)
SCALE = 4                  # 4-month index ending September = JJAS monsoon
GROUP_NAMES = {1: "Irrigated", 2: "Rainfed"}
COLORS = {"Irrigated": "#1f78b4", "Rainfed": "#d95f02"}

OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
summary = []


def say(text=""):
    print(text)
    summary.append(text)


# ---------------- 1. CLIMATE: PET, SPI, SPEI ----------------
def hargreaves_pet(df, lat_deg):
    """Monthly PET (mm) by Hargreaves-Samani (FAO-56 eq. 52), Ra from FAO-56 eq. 21."""
    J = (df.index + pd.offsets.Day(14)).dayofyear.to_numpy()          # mid-month day
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec)
                                          + np.cos(phi) * np.cos(dec) * np.sin(ws))
    tmean = (df.tmax_C + df.tmin_C) / 2
    trange = (df.tmax_C - df.tmin_C).clip(lower=0)
    pet_day = 0.0023 * 0.408 * ra * (tmean + 17.8) * np.sqrt(trange)
    return pet_day * df.index.days_in_month


def spi_from_series(x):
    """Gamma-based SPI for one calendar month's series (handles zeros)."""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    ok = ~np.isnan(x)
    v = x[ok]
    q = np.mean(v <= 0)
    pos = v[v > 0]
    a, loc, sc = stats.gamma.fit(pos, floc=0)
    H = q + (1 - q) * stats.gamma.cdf(np.maximum(v, 0), a, loc=loc, scale=sc)
    out[ok] = stats.norm.ppf(np.clip(H, 1e-6, 1 - 1e-6))
    return out


def spei_from_series(d):
    """3-parameter log-logistic SPEI fitted by probability-weighted moments
    (Vicente-Serrano et al., 2010, J. Climate 23:1696-1718)."""
    from scipy.special import gamma as G
    d = np.asarray(d, float)
    out = np.full_like(d, np.nan)
    ok = ~np.isnan(d)
    x = np.sort(d[ok])
    n = len(x)
    F = (np.arange(1, n + 1) - 0.35) / n
    w0, w1, w2 = [np.mean((1 - F) ** s * x) for s in range(3)]
    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    g = G(1 + 1 / beta) * G(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / g
    gam = w0 - alpha * g
    if np.isfinite(alpha) and alpha > 0 and beta > 1:
        z = np.maximum(d[ok] - gam, 1e-9)
        cdf = 1 / (1 + (alpha / z) ** beta)
    else:  # fit failed (rare, very skewed month): Gringorten empirical probability
        cdf = (stats.rankdata(d[ok]) - 0.44) / (n + 0.12)
    out[ok] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


def standardized_index(series, scale, func):
    acc = series.rolling(scale).sum()
    res = pd.Series(np.nan, index=series.index)
    for m in range(1, 13):
        idx = acc.index[(acc.index.month == m) & acc.notna()]
        res.loc[idx] = func(acc.loc[idx].to_numpy())
    return res


clim = pd.read_csv(DATA_DIR / "gaya_climate_monthly.csv")
clim.index = pd.to_datetime(clim["date"] + "-01")
clim = clim.drop(columns="date").sort_index()
clim["pet_mm"] = hargreaves_pet(clim, LAT_DEG)
clim["wb_mm"] = clim.pr_mm - clim.pet_mm
clim[f"SPI{SCALE}"] = standardized_index(clim.pr_mm, SCALE, spi_from_series)
clim[f"SPEI{SCALE}"] = standardized_index(clim.wb_mm, SCALE, spei_from_series)
clim.to_csv(OUT_DIR / "climate_monthly_with_SPI_SPEI.csv", index_label="date")

sep = clim[clim.index.month == 9]
annual = pd.DataFrame({
    "year": sep.index.year,
    "JJAS_rain_mm": clim.pr_mm.rolling(4).sum()[clim.index.month == 9].to_numpy(),
    "SPI": sep[f"SPI{SCALE}"].to_numpy(),
    "SPEI": sep[f"SPEI{SCALE}"].to_numpy(),
}).set_index("year")
jfm = clim[clim.index.month <= 3]
rabi_tmax = jfm.groupby(jfm.index.year).tmax_C.mean()          # Jan-Mar of year y+1
annual["rabi_tmax_C"] = rabi_tmax.reindex(annual.index + 1).to_numpy()
annual["drought"] = annual.SPEI <= DROUGHT_T
annual.to_csv(OUT_DIR / "annual_drought_indices.csv")

say("=" * 70)
say("DROUGHT BUFFERING BY IRRIGATION - GAYA DISTRICT, BIHAR")
say("=" * 70)
say(f"Monsoon (JJAS) rainfall {annual.index.min()}-{annual.index.max()}: "
    f"mean {annual.JJAS_rain_mm.mean():.0f} mm, CV {annual.JJAS_rain_mm.std() / annual.JJAS_rain_mm.mean() * 100:.0f}%")
dyears = annual.index[annual.drought].tolist()
say(f"Drought years (SPEI-{SCALE} Sep <= {DROUGHT_T}): {dyears}")


# ---------------- 2. TRENDS (Mann-Kendall + Sen's slope) ----------------
def mann_kendall(y):
    y = np.asarray(pd.Series(y).dropna(), float)
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slope = stats.theilslopes(y, np.arange(n))[0]
    return s, z, p, slope


trend_rows = []


def trend(name, series):
    s, z, p, slope = mann_kendall(series)
    trend_rows.append(dict(variable=name, n=series.notna().sum(), S=s, Z=round(z, 3),
                           p_value=round(p, 4), sen_slope_per_decade=round(slope * 10, 4)))


trend("JJAS rainfall (mm)", annual.JJAS_rain_mm)
trend(f"SPEI-{SCALE} (Sep)", annual.SPEI)
trend("Rabi Jan-Mar Tmax (C)", annual.rabi_tmax_C)

# ---------------- 3. VHI BY GROUP ----------------
vhi = pd.read_csv(DATA_DIR / "gaya_vhi_by_group.csv")
vhi = vhi[vhi.grp.isin(GROUP_NAMES)]
vhi["group"] = vhi.grp.map(GROUP_NAMES)
vhi = vhi.merge(annual[["SPEI", "SPI", "drought", "rabi_tmax_C"]], left_on="year",
                right_index=True, how="left")
vhi["condition"] = np.where(vhi.drought, "Drought", "Normal")

# Remove the long-term greening trend (intensification, new varieties, irrigation growth)
# per season and group, so drought years early and late in the record are comparable.
# VHI below = trend-removed value + the group's mean; VHI_raw keeps the original.
DETREND = True
vhi["VHI_raw"] = vhi.VHI
if DETREND:
    for (season, g), idx in vhi.groupby(["season", "group"]).groups.items():
        d = vhi.loc[idx].dropna(subset=["VHI_raw"])
        if len(d) > 5:
            fit = np.polyfit(d.year, d.VHI_raw, 1)
            vhi.loc[d.index, "VHI"] = d.VHI_raw - np.polyval(fit, d.year) + d.VHI_raw.mean()
vhi.to_csv(OUT_DIR / "vhi_merged.csv", index=False)

for season in ["kharif", "rabi"]:
    for g in GROUP_NAMES.values():
        s = vhi[(vhi.season == season) & (vhi.group == g)].set_index("year").sort_index().VHI_raw
        trend(f"VHI {season} - {g}", s)


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return beta, se, 2 * (1 - stats.t.cdf(np.abs(t), dof))


buffer_rows = []
for season in ["kharif", "rabi"]:
    d = vhi[vhi.season == season].dropna(subset=["VHI", "SPEI"])
    if d.empty:
        continue
    say("")
    say(f"--- {season.upper()} season "
        f"({'Aug 15-Oct 31' if season == 'kharif' else 'Jan 1-Mar 15, after the monsoon of year y'}) ---")
    m = d.groupby(["group", "condition"]).VHI.mean().unstack()
    if {"Drought", "Normal"} <= set(m.columns):
        impact = m["Normal"] - m["Drought"]
        # only meaningful when rainfed crops were actually hurt (drop of at least 2 VHI points)
        bi = 1 - impact["Irrigated"] / impact["Rainfed"] if impact["Rainfed"] >= 2 else np.nan
        for g in m.index:
            say(f"{g:9s}: VHI normal {m.loc[g, 'Normal']:.1f}, drought {m.loc[g, 'Drought']:.1f}, "
                f"drop {impact[g]:.1f} points")
        say(f"Buffering index (share of rainfed drought impact avoided by irrigation) = {bi:.2f}")
        gap = d.pivot_table(index="year", columns="group", values="VHI")
        gap = (gap["Irrigated"] - gap["Rainfed"]).to_frame("gap").join(annual.drought)
        g_d, g_n = gap.gap[gap.drought], gap.gap[~gap.drought]
        u = stats.mannwhitneyu(g_d, g_n, alternative="greater") if len(g_d) > 1 else None
        say(f"Irrigated-minus-rainfed VHI gap: drought years {g_d.mean():.1f}, normal years "
            f"{g_n.mean():.1f}" + (f" (Mann-Whitney U one-sided p = {u.pvalue:.3f})" if u else ""))
    else:
        bi = np.nan
        say("Not enough drought and normal years to compare.")

    # Sensitivity: VHI = b0 + b1*SPEI + b2*Rainfed + b3*SPEI*Rainfed
    rain = (d.group == "Rainfed").astype(float).to_numpy()
    sp = d.SPEI.to_numpy()
    beta, se, p = ols(np.column_stack([sp, rain, sp * rain]), d.VHI.to_numpy())
    say(f"VHI sensitivity to SPEI: irrigated {beta[1]:.2f}/unit (p={p[1]:.3f}); "
        f"rainfed {beta[1] + beta[3]:.2f}/unit; difference {beta[3]:.2f} (interaction p={p[3]:.3f})")
    for g in GROUP_NAMES.values():
        dg = d[d.group == g]
        lr = stats.linregress(dg.SPEI, dg.VHI)
        buffer_rows.append(dict(season=season, group=g, n_years=len(dg),
                                VHI_normal=dg[~dg.drought].VHI.mean(), VHI_drought=dg[dg.drought].VHI.mean(),
                                slope_VHI_per_SPEI=lr.slope, r=lr.rvalue, p_value=lr.pvalue,
                                buffering_index=bi))
pd.DataFrame(buffer_rows).round(4).to_csv(OUT_DIR / "buffering_results.csv", index=False)

# ---------------- 4. PIXEL SENSITIVITY vs IRRIGATION ----------------
pix_file = DATA_DIR / "gaya_pixel_sensitivity.csv"
pix = None
if pix_file.exists():
    pix = pd.read_csv(pix_file).dropna(subset=["slope", "irr_index"])
    rho, prho = stats.spearmanr(pix.irr_index, pix.slope)
    say("")
    say("--- PIXEL LEVEL (Kharif VHI vs standardized JJAS rainfall) ---")
    say(f"{len(pix)} cropland pixels; Spearman rho(irrigation index, sensitivity) = {rho:.2f} (p = {prho:.2g})")
    gi, gr = pix[pix.grp == 1].slope, pix[pix.grp == 2].slope
    if len(gi) > 5 and len(gr) > 5:
        u = stats.mannwhitneyu(gi, gr, alternative="less")
        say(f"Median sensitivity: irrigated {gi.median():.2f}, rainfed {gr.median():.2f} VHI points per SD "
            f"of rainfall (Mann-Whitney p = {u.pvalue:.2g})")
    pix["irr_bin"] = pd.cut(pix.irr_index, np.linspace(0, 1, 11), include_lowest=True)
    binned = pix.groupby("irr_bin", observed=True).slope.agg(["mean", "sem", "count"])
    binned.to_csv(OUT_DIR / "sensitivity_by_irrigation_bin.csv")

# ---------------- 5. IRRIGATED AREA (Script 1) ----------------
area_file = DATA_DIR / "gaya_irrigated_area_by_year.csv"
area = None
if area_file.exists():
    area = pd.read_csv(area_file).sort_values("year")
    area = area.rename(columns={"ha_class0": "Irrigated", "ha_class1": "Rainfed", "ha_class2": "Fallow"})
    say("")
    say("--- IRRIGATED AREA (Sentinel-2) ---")
    for _, r in area.iterrows():
        tot = r[["Irrigated", "Rainfed", "Fallow"]].sum()
        say(f"{int(r.year)}-{str(int(r.year) + 1)[2:]} ({r.method}): irrigated {r.Irrigated / 1e3:.1f} kha "
            f"({r.Irrigated / tot * 100:.0f}% of cropland)")
    if len(area) >= 4:
        trend("Irrigated area (ha)", area.set_index("year").Irrigated)

tr = pd.DataFrame(trend_rows)
tr.to_csv(OUT_DIR / "trend_tests.csv", index=False)
say("")
say("--- TRENDS (Mann-Kendall, Sen's slope per decade) ---")
for _, r in tr.iterrows():
    star = "*" if r.p_value < 0.05 else " "
    say(f"{r.variable:32s} Z={r.Z:6.2f}  p={r.p_value:.3f}{star}  Sen={r.sen_slope_per_decade:+.3f}/decade")

# ---------------- 6. FIGURES ----------------
def shade_droughts(ax):
    for y in dyears:
        ax.axvspan(y - 0.5, y + 0.5, color="0.85", zorder=0, lw=0)


# Fig 1 - SPEI and monsoon rainfall
fig, ax = plt.subplots(figsize=(7.5, 3))
cols = np.where(annual.SPEI < 0, "#d95f02", "#1f78b4")
ax.bar(annual.index, annual.SPEI, color=cols, width=0.8)
ax.axhline(DROUGHT_T, color="k", ls="--", lw=0.8)
ax.text(annual.index.min(), DROUGHT_T - 0.25, "drought threshold", fontsize=8)
ax.set_ylabel(f"SPEI-{SCALE} (Sep)")
ax.set_title("Monsoon water balance, Gaya district")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_spei_timeseries.png")

# Fig 2 - VHI time series by group
fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True)
for ax, season in zip(axes, ["kharif", "rabi"]):
    shade_droughts(ax)
    for g in GROUP_NAMES.values():
        s = vhi[(vhi.season == season) & (vhi.group == g)].sort_values("year")
        ax.plot(s.year, s.VHI, "-o", ms=3, color=COLORS[g], label=g)
    ax.set_ylabel(f"VHI ({season.capitalize()})")
    ax.set_xlim(vhi.year.min() - 0.7, vhi.year.max() + 0.7)
axes[0].legend(frameon=False, ncol=2)
axes[0].set_title("Vegetation Health Index (shaded = drought years)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_vhi_timeseries.png")

# Fig 3 - boxplots drought vs normal
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
for ax, season in zip(axes, ["kharif", "rabi"]):
    data, labels, colors = [], [], []
    for g in GROUP_NAMES.values():
        for c in ["Normal", "Drought"]:
            data.append(vhi[(vhi.season == season) & (vhi.group == g) & (vhi.condition == c)].VHI.dropna())
            labels.append(f"{g}\n{c}")
            colors.append(COLORS[g])
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, c, lab in zip(bp["boxes"], colors, labels):
        patch.set_facecolor(c)
        patch.set_alpha(0.35 if "Normal" in lab else 0.85)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.set_title(season.capitalize())
axes[0].set_ylabel("VHI")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_vhi_boxplots.png")

# Fig 4 - VHI vs SPEI regression
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
for ax, season in zip(axes, ["kharif", "rabi"]):
    for g in GROUP_NAMES.values():
        dg = vhi[(vhi.season == season) & (vhi.group == g)].dropna(subset=["SPEI", "VHI"])
        ax.scatter(dg.SPEI, dg.VHI, s=14, color=COLORS[g], alpha=0.8, label=g)
        lr = stats.linregress(dg.SPEI, dg.VHI)
        xs = np.linspace(dg.SPEI.min(), dg.SPEI.max(), 10)
        ax.plot(xs, lr.intercept + lr.slope * xs, color=COLORS[g])
        ax.text(0.03, 0.95 - 0.08 * list(GROUP_NAMES.values()).index(g),
                f"{g}: slope {lr.slope:.1f}, r={lr.rvalue:.2f}", transform=ax.transAxes,
                fontsize=8, color=COLORS[g], va="top")
    ax.set_xlabel(f"SPEI-{SCALE} (monsoon)")
    ax.set_title(season.capitalize())
axes[0].set_ylabel("VHI")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_vhi_vs_spei.png")

# Fig 5 - pixel sensitivity vs irrigation index
if pix is not None:
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    mids = [iv.mid for iv in binned.index]
    ax.errorbar(mids, binned["mean"], yerr=1.96 * binned["sem"], fmt="o-", color="#1f78b4", capsize=3)
    ax.set_xlabel("Irrigation index (0 = rainfed, 1 = fully irrigated)")
    ax.set_ylabel("VHI sensitivity to rainfall\n(VHI points per SD)")
    ax.set_title("Drought sensitivity falls with irrigation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_sensitivity_vs_irrigation.png")

# Fig 6 - irrigated area by year
if area is not None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bottom = np.zeros(len(area))
    for cls, col in [("Irrigated", "#1f78b4"), ("Rainfed", "#b2df8a"), ("Fallow", "#bdbdbd")]:
        vals = area[cls].fillna(0).to_numpy() / 1e3
        ax.bar(area.year.astype(str), vals, bottom=bottom, color=col, label=cls)
        bottom += vals
    ax.set_ylabel("Area (thousand ha)")
    ax.set_title("Cropland classes, Gaya (Sentinel-2)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_irrigated_area.png")

(OUT_DIR / "results_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print(f"\nDone. Tables, figures and results_summary.txt are in: {OUT_DIR}")
