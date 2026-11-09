"""
SCRIPT 7 - Novelty analyses for the Gaya drought-buffering paper

  A. Convergence of evidence: agreement between 3 independent irrigation sources
  B. Legacy and compound drought: does irrigation still buffer when a failed
     monsoon is followed by a hot season?
  C. "Buffering at what cost?": groundwater storage from GRACE/GRACE-FO
  D. Explainable machine learning: what controls drought sensitivity?

Inputs
  ./outputs  (from Script 3): annual_drought_indices.csv, vhi_merged.csv
  ./data     (GEE exports):   gaya_climate_monthly.csv          (Script 2)
                              gaya_pixel_sensitivity.csv        (Script 2, with drivers)
                              gaya_agreement_area.csv           (Script 5)
                              gaya_agreement_samples.csv        (Script 5)
                              gaya_grace_monthly.csv            (Script 6)
Every section runs only if its input files exist.

Run:
    pip install pandas numpy scipy matplotlib scikit-learn
    pip install shap            (optional: SHAP plots; otherwise permutation importance)
    python 07_novelty_analysis.py
Outputs: ./outputs_novelty  (tables, figures, novelty_summary.txt)
"""
import json
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
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "outputs_novelty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROUGHT_T = -1.0          # SPEI threshold (same as Script 3)
HEAT_Z = 1.0              # hot season = Tmax anomaly >= 1 standard deviation
GRACE_BASE = (2004, 2009)  # GRACE convention baseline for anomalies
COLORS = {"Irrigated": "#1f78b4", "Rainfed": "#d95f02"}
plt.rcParams.update({"font.size": 10, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
summary = []


def say(t=""):
    print(t)
    summary.append(t)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    cov = resid @ resid / dof * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    p = 2 * (1 - stats.t.cdf(np.abs(beta / se), dof))
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return beta, se, p, r2


def mann_kendall(y):
    y = np.asarray(pd.Series(y).dropna(), float)
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, c = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var)
    return z, 2 * (1 - stats.norm.cdf(abs(z))), stats.theilslopes(y, np.arange(n))[0]


def exists(*names, folder=None):
    folder = folder or DATA_DIR
    return all((folder / n).exists() for n in names)


say("=" * 74)
say("NOVELTY ANALYSES - GAYA DROUGHT BUFFERING STUDY")
say("=" * 74)

annual = None
if exists("annual_drought_indices.csv", folder=OUT3_DIR):
    annual = pd.read_csv(OUT3_DIR / "annual_drought_indices.csv").set_index("year")
    dyears = annual.index[annual.drought.astype(bool)].tolist()
else:
    dyears = []
    say("NOTE: run Script 3 first (outputs/annual_drought_indices.csv) for sections B and C.")

# =====================================================================
# A. CONVERGENCE OF EVIDENCE
# =====================================================================
if exists("gaya_agreement_samples.csv"):
    from sklearn.metrics import cohen_kappa_score
    say("")
    say("--- A. AGREEMENT BETWEEN 3 INDEPENDENT IRRIGATION SOURCES ---")
    s = pd.read_csv(DATA_DIR / "gaya_agreement_samples.csv")
    src = {"Sentinel-2 RF (Rabi-cropped)": "s2_rabi", "WorldCereal irrigated": "wc_irr",
           "MODIS persistent Rabi green": "modis_rabi"}
    s = s.dropna(subset=list(src.values()))
    rows = []
    names = list(src)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = s[src[names[i]]], s[src[names[j]]]
            rows.append(dict(source_1=names[i], source_2=names[j],
                             percent_agreement=round((a == b).mean() * 100, 1),
                             cohen_kappa=round(cohen_kappa_score(a, b), 3)))
    pair = pd.DataFrame(rows)
    pair.to_csv(OUT_DIR / "A_pairwise_agreement.csv", index=False)
    # Fleiss' kappa for 3 binary raters
    X = s[list(src.values())].to_numpy(int)
    n_r = X.shape[1]
    counts = np.column_stack([n_r - X.sum(1), X.sum(1)])
    P_i = (np.sum(counts ** 2, 1) - n_r) / (n_r * (n_r - 1))
    p_j = counts.sum(0) / counts.sum()
    fleiss = (P_i.mean() - np.sum(p_j ** 2)) / (1 - np.sum(p_j ** 2))
    for r in rows:
        say(f"{r['source_1']} vs {r['source_2']}: agreement {r['percent_agreement']}%, "
            f"kappa {r['cohen_kappa']}")
    say(f"Fleiss' kappa (all three sources): {fleiss:.3f}   (n = {len(s)} cropland pixels)")
    for k, c in src.items():
        say(f"  share of cropland called irrigated by {k}: {s[c].mean() * 100:.0f}%")

if exists("gaya_agreement_area.csv"):
    ag = pd.read_csv(DATA_DIR / "gaya_agreement_area.csv").sort_values("score")
    ag["share_pct"] = ag.area_ha / ag.area_ha.sum() * 100
    ag.round(1).to_csv(OUT_DIR / "A_area_by_agreement_score.csv", index=False)
    cons = ag[ag.score >= 2].area_ha.sum()
    core = ag[ag.score == 3].area_ha.sum()
    say(f"Consensus irrigated area (at least 2 of 3 sources): {cons:,.0f} ha; "
        f"all 3 sources: {core:,.0f} ha")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    cols = ["#d9d9d9", "#fdd49e", "#fc8d59", "#0570b0"]
    ax.bar(ag.score.astype(int).astype(str), ag.area_ha / 1e3,
           color=[cols[int(v)] for v in ag.score])
    ax.set_xlabel("Number of sources saying 'irrigated'")
    ax.set_ylabel("Area (thousand ha)")
    ax.set_title("Convergence of evidence, Gaya cropland")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_agreement_area.png")

# =====================================================================
# B. LEGACY AND COMPOUND DROUGHT
# =====================================================================
if annual is not None and exists("vhi_merged.csv", folder=OUT3_DIR) and exists("gaya_climate_monthly.csv"):
    say("")
    say("--- B. LEGACY AND COMPOUND DROUGHT (does buffering hold under dry + hot?) ---")
    vhi = pd.read_csv(OUT3_DIR / "vhi_merged.csv")
    vhi = vhi[vhi.group.isin(COLORS)]
    cm = pd.read_csv(DATA_DIR / "gaya_climate_monthly.csv")
    cm["year"] = cm.date.str[:4].astype(int)
    cm["month"] = cm.date.str[5:7].astype(int)
    heat_kh = cm[cm.month.isin([8, 9, 10])].groupby("year").tmax_C.mean()       # Aug–Oct, year y
    heat_rb = cm[cm.month.isin([2, 3])].groupby("year").tmax_C.mean()            # Feb–Mar, year y+1
    heat = {"kharif": heat_kh, "rabi": heat_rb.rename(lambda y: y - 1)}          # align to agri year
    rows, coef_rows = [], []
    for season in ["kharif", "rabi"]:
        d = vhi[vhi.season == season].copy()
        h = heat[season]
        d["heat_z"] = d.year.map((h - h.mean()) / h.std())
        d = d.dropna(subset=["VHI", "SPEI", "heat_z"])
        dry, hot = d.SPEI <= DROUGHT_T, d.heat_z >= HEAT_Z
        d["category"] = np.select([dry & hot, dry, hot], ["Compound (dry + hot)", "Dry only", "Hot only"],
                                  "Normal")
        tab = d.groupby(["category", "group"]).VHI.agg(["mean", "count"]).unstack("group")
        say(f"{season.upper()}: mean VHI by year type (n years in brackets)")
        for cat in ["Normal", "Dry only", "Hot only", "Compound (dry + hot)"]:
            if cat in tab.index:
                parts = [f"{g} {tab.loc[cat, ('mean', g)]:.1f} ({int(tab.loc[cat, ('count', g)])})"
                         for g in COLORS if ('mean', g) in tab.columns and pd.notna(tab.loc[cat, ('mean', g)])]
                say(f"   {cat:22s}: " + ", ".join(parts))
                for g in COLORS:
                    if ('mean', g) in tab.columns and pd.notna(tab.loc[cat, ('mean', g)]):
                        rows.append(dict(season=season, category=cat, group=g,
                                         VHI_mean=tab.loc[cat, ('mean', g)],
                                         n_years=int(tab.loc[cat, ('count', g)])))
        for g in COLORS:
            dg = d[d.group == g]
            if len(dg) < 8:
                continue
            X = np.column_stack([dg.SPEI, dg.heat_z, dg.SPEI * dg.heat_z])
            b, se, p, r2 = ols(X, dg.VHI.to_numpy())
            label = "monsoon SPEI (legacy)" if season == "rabi" else "monsoon SPEI"
            say(f"   {g:9s} VHI = {b[0]:.1f} + {b[1]:+.2f}*SPEI (p={p[1]:.3f}) {b[2]:+.2f}*heat (p={p[2]:.3f}) "
                f"{b[3]:+.2f}*SPEI x heat (p={p[3]:.3f}), R2={r2:.2f}")
            coef_rows.append(dict(season=season, group=g, b_SPEI=b[1], p_SPEI=p[1], b_heat=b[2], p_heat=p[2],
                                  b_interaction=b[3], p_interaction=p[3], R2=r2, n=len(dg), note=label))
        # FORMAL TEST: does the irrigation advantage grow with heat / drought?
        # Paired by year: gap = VHI(irrigated) - VHI(rainfed) = a + c1*SPEI + c2*heat.
        # c2 > 0 -> irrigated fields lose less than rainfed fields as heat rises (heat buffering);
        # c1 < 0 -> the gap widens in dry years (drought buffering). Bootstrap CI over years.
        pv = d.pivot_table(index="year", columns="group", values="VHI")
        if {"Irrigated", "Rainfed"} <= set(pv.columns):
            gp = (pv.Irrigated - pv.Rainfed).rename("gap").to_frame().join(
                d.drop_duplicates("year").set_index("year")[["SPEI", "heat_z"]]).dropna()
            Xg = np.column_stack([gp.SPEI, gp.heat_z])
            bg, seg, pg, r2g = ols(Xg, gp.gap.to_numpy())
            rng = np.random.default_rng(42)
            boots = []
            for _ in range(2000):
                ii = rng.integers(0, len(gp), len(gp))
                try:
                    boots.append(ols(Xg[ii], gp.gap.to_numpy()[ii])[0][1:])
                except np.linalg.LinAlgError:
                    pass
            ci = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
            say(f"   FORMAL TEST ({season}, n={len(gp)} years): irrigated-minus-rainfed gap = {bg[0]:.1f} "
                f"{bg[1]:+.2f}*SPEI (p={pg[1]:.3f}, 95% CI {ci[0, 0]:+.2f} to {ci[1, 0]:+.2f}) "
                f"{bg[2]:+.2f}*heat (p={pg[2]:.3f}, 95% CI {ci[0, 1]:+.2f} to {ci[1, 1]:+.2f})")
            coef_rows.append(dict(season=season, group="gap (irrigated - rainfed)", b_SPEI=bg[1], p_SPEI=pg[1],
                                  b_heat=bg[2], p_heat=pg[2], b_interaction=np.nan, p_interaction=np.nan,
                                  R2=r2g, n=len(gp), note=f"bootstrap 95% CI SPEI [{ci[0,0]:.2f},{ci[1,0]:.2f}] "
                                                          f"heat [{ci[0,1]:.2f},{ci[1,1]:.2f}]"))
    comp = pd.DataFrame(rows)
    comp.round(3).to_csv(OUT_DIR / "B_compound_vhi.csv", index=False)
    pd.DataFrame(coef_rows).round(4).to_csv(OUT_DIR / "B_compound_regression.csv", index=False)
    say("   (Rabi SPEI = the PREVIOUS monsoon: a significant slope means a legacy effect.)")
    if not comp.empty:
        cats = ["Normal", "Dry only", "Hot only", "Compound (dry + hot)"]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
        for ax, season in zip(axes, ["kharif", "rabi"]):
            c = comp[comp.season == season]
            x = np.arange(len(cats))
            for k, g in enumerate(COLORS):
                cg = c[c.group == g].set_index("category").reindex(cats)
                ax.bar(x + (k - 0.5) * 0.38, cg.VHI_mean, 0.38, color=COLORS[g], label=g)
            ax.set_xticks(x, ["Normal", "Dry", "Hot", "Dry+hot"])
            ax.set_title(season.capitalize())
        axes[0].set_ylabel("Mean VHI")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig8_compound_drought.png")

# =====================================================================
# C. GROUNDWATER COST (GRACE)
# =====================================================================
if exists("gaya_grace_monthly.csv"):
    say("")
    say("--- C. BUFFERING AT WHAT COST? GROUNDWATER STORAGE (GRACE/GRACE-FO) ---")
    g = pd.read_csv(DATA_DIR / "gaya_grace_monthly.csv")
    g.index = pd.to_datetime(g.date + "-01")
    base = (g.index.year >= GRACE_BASE[0]) & (g.index.year <= GRACE_BASE[1])
    for c in ["sm_mm", "canopy_mm", "swe_mm", "gws_clsm_mm"]:
        g[c + "_anom"] = g[c] - g.loc[base, c].mean()
    g["TWSA_mm"] = g.tws_cm * 10
    g["GWSA_grace_mm"] = g.TWSA_mm - g.sm_mm_anom - g.canopy_mm_anom - g.swe_mm_anom.fillna(0)
    g["GWSA_clsm_mm"] = g.gws_clsm_mm_anom
    n_grace = g.GWSA_grace_mm.notna().sum()
    use = "GWSA_grace_mm" if n_grace >= 120 else "GWSA_clsm_mm"
    say(f"GRACE months available: {n_grace}; primary series: "
        f"{'GRACE TWS minus GLDAS soil moisture' if use == 'GWSA_grace_mm' else 'GLDAS-2.2 CLSM (GRACE-assimilated)'}")
    if g.GWSA_grace_mm.notna().sum() > 24 and g.GWSA_clsm_mm.notna().sum() > 24:
        both = g[["GWSA_grace_mm", "GWSA_clsm_mm"]].dropna()
        say(f"Agreement of the two groundwater estimates: r = {both.corr().iloc[0, 1]:.2f} (n = {len(both)} months)")
    g[["TWSA_mm", "GWSA_grace_mm", "GWSA_clsm_mm"]].round(2).to_csv(OUT_DIR / "C_groundwater_monthly.csv",
                                                                   index_label="month")

    series = g[use]

    def month_val(y, m):
        t = pd.Timestamp(year=y, month=m, day=1)
        if t in series.index and pd.notna(series.get(t)):
            return series[t]
        nb = [series.get(t + pd.DateOffset(months=d)) for d in (-1, 1)]
        nb = [v for v in nb if v is not None and pd.notna(v)]
        return np.mean(nb) if nb else np.nan

    yrs = range(series.index.year.min(), series.index.year.max())
    seas = pd.DataFrame({
        "year": list(yrs),
        "recharge_mm": [month_val(y, 10) - month_val(y, 5) for y in yrs],          # monsoon gain May->Oct
        "drawdown_mm": [month_val(y + 1, 5) - month_val(y, 10) for y in yrs],      # Rabi+summer Oct->May
    }).set_index("year")
    ann = series.groupby(series.index.year).agg(["mean", "count"])
    ann = ann[ann["count"] >= 8]["mean"]
    z, p, _ = mann_kendall(ann)
    slope = stats.theilslopes(ann.to_numpy(), ann.index.to_numpy())[0]   # per YEAR (series has a GRACE gap)
    say(f"Annual groundwater storage anomaly trend: {slope * 10:+.0f} mm per decade "
        f"(Mann-Kendall p = {p:.3f}, {ann.index.min()}-{ann.index.max()})")
    if annual is not None:
        seas = seas.join(annual[["SPEI", "JJAS_rain_mm", "drought"]], how="left")
        seas["drought"] = seas.drought.fillna(False).astype(bool)
        sd = seas.dropna(subset=["recharge_mm", "drawdown_mm"])
        dr, nm = sd[sd.drought], sd[~sd.drought]
        if len(dr) and len(nm):
            say(f"Monsoon recharge: drought years {dr.recharge_mm.mean():+.0f} mm vs normal {nm.recharge_mm.mean():+.0f} mm")
            say(f"Post-monsoon drawdown (Oct->May): drought years {dr.drawdown_mm.mean():+.0f} mm vs "
                f"normal {nm.drawdown_mm.mean():+.0f} mm")
        if len(sd) > 5:
            lr = stats.linregress(sd.SPEI, sd.drawdown_mm)
            say(f"Drawdown vs monsoon SPEI: slope {lr.slope:+.1f} mm per SPEI unit (r = {lr.rvalue:.2f}, p = {lr.pvalue:.3f})")
        # cost per unit of buffering (Kharif VHI gap irrigated - rainfed)
        if exists("vhi_merged.csv", folder=OUT3_DIR) and len(dr) and len(nm):
            v = pd.read_csv(OUT3_DIR / "vhi_merged.csv")
            gap = v[v.season == "kharif"].pivot_table(index="year", columns="group", values="VHI")
            if {"Irrigated", "Rainfed"} <= set(gap.columns):
                gap = (gap.Irrigated - gap.Rainfed).rename("gap")
                cost = sd.join(gap, how="inner")
                normal_gap = cost[~cost.drought].gap.mean()
                normal_dd = cost[~cost.drought].drawdown_mm.mean()
                cd = cost[cost.drought].copy()
                cd["extra_buffering_VHI"] = cd.gap - normal_gap
                cd["extra_drawdown_mm"] = cd.drawdown_mm - normal_dd
                cd["mm_per_VHI_point"] = -cd.extra_drawdown_mm / cd.extra_buffering_VHI.replace(0, np.nan)
                cd.round(2).to_csv(OUT_DIR / "C_buffering_cost_drought_years.csv")
                for y, r in cd.iterrows():
                    say(f"   drought {y}: extra buffering {r.extra_buffering_VHI:+.1f} VHI points, "
                        f"extra groundwater drawdown {r.extra_drawdown_mm:+.0f} mm")
        seas.round(2).to_csv(OUT_DIR / "C_groundwater_seasonal.csv")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for y in dyears:
        if g.index.min().year <= y <= g.index.max().year:
            ax.axvspan(pd.Timestamp(y, 6, 1), pd.Timestamp(y + 1, 5, 31), color="0.88", lw=0, zorder=0)
    ax.set_xlim(g.index.min(), g.index.max())
    ax.plot(g.index, g.GWSA_grace_mm, color="#1f78b4", lw=1, label="GRACE TWS - GLDAS soil moisture")
    ax.plot(g.index, g.GWSA_clsm_mm, color="#d95f02", lw=1, alpha=0.8, label="GLDAS-2.2 CLSM (GRACE-DA)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Groundwater storage\nanomaly (mm)")
    ax.set_title("Groundwater storage, Gaya region (shaded = drought years)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_groundwater_timeseries.png")

    sd2 = seas.dropna(subset=["drawdown_mm"])
    if len(sd2):
        fig, ax = plt.subplots(figsize=(8, 3))
        col = ["#d95f02" if (annual is not None and bool(r.get("drought", False))) else "#9ecae1"
               for _, r in sd2.iterrows()]
        ax.bar(sd2.index, sd2.drawdown_mm, color=col)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Oct -> May change (mm)")
        ax.set_title("Post-monsoon groundwater drawdown (orange = drought year)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig10_groundwater_drawdown.png")

# =====================================================================
# D. EXPLAINABLE MACHINE LEARNING: DRIVERS OF DROUGHT SENSITIVITY
# =====================================================================
if exists("gaya_pixel_sensitivity.csv"):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay
    from sklearn.metrics import r2_score

    pix = pd.read_csv(DATA_DIR / "gaya_pixel_sensitivity.csv")
    candidates = ["irr_index", "wc_irr_frac", "elev_m", "slope_deg", "dist_river_m",
                  "clay_pct", "rain_cv", "mar_tmax_C"]
    feats = [c for c in candidates if c in pix.columns and pix[c].notna().mean() > 0.8]
    if len(feats) >= 3:
        say("")
        say("--- D. WHAT CONTROLS DROUGHT SENSITIVITY? (Random Forest + spatial cross-validation) ---")
        d = pix.dropna(subset=feats + ["slope"]).copy()
        coords = d[".geo"].apply(lambda s: json.loads(s)["coordinates"] if isinstance(s, str) and s.startswith("{")
                                 else [np.nan, np.nan])
        d["lon"] = coords.str[0]
        d["lat"] = coords.str[1]
        blocks = (np.floor(d.lon.fillna(0) / 0.1).astype(int).astype(str) + "_" +
                  np.floor(d.lat.fillna(0) / 0.1).astype(int).astype(str))
        X, y = d[feats].astype(float), d.slope.astype(float)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=1)
        pred = np.full(len(y), np.nan)
        n_groups = blocks.nunique()
        gkf = GroupKFold(n_splits=min(5, max(2, n_groups)))
        fold_imp = []
        for tr, te in gkf.split(X, y, blocks):
            rf.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = rf.predict(X.iloc[te])
            # importance on the HELD-OUT spatial blocks (not on training data)
            fold_imp.append(permutation_importance(rf, X.iloc[te], y.iloc[te], n_repeats=5,
                                                   random_state=1, n_jobs=-1).importances_mean)
        r2cv = r2_score(y, pred)
        say(f"{len(d)} pixels, {len(feats)} drivers; spatial-block cross-validated R2 = {r2cv:.2f} "
            f"({n_groups} blocks of 0.1 deg)")
        fold_imp = np.array(fold_imp)
        rf.fit(X, y)
        imp = pd.DataFrame({"driver": feats, "importance": fold_imp.mean(0),
                            "sd": fold_imp.std(0)}).sort_values("importance", ascending=False)
        try:
            import shap
            samp = X.sample(min(1500, len(X)), random_state=1)
            sv = shap.TreeExplainer(rf).shap_values(samp)
            imp["mean_abs_shap"] = imp.driver.map(dict(zip(feats, np.abs(sv).mean(0))))
            shap.summary_plot(sv, samp, show=False)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "fig12_shap_summary.png")
            plt.close()
            method = "SHAP"
        except ImportError:
            method = "permutation importance (install 'shap' for SHAP plots)"
        imp.round(4).to_csv(OUT_DIR / "D_driver_importance.csv", index=False)
        say(f"Driver ranking ({method}):")
        for _, r in imp.iterrows():
            rho = stats.spearmanr(d[r.driver], d.slope)[0]
            say(f"   {r.driver:13s} importance {r.importance:.3f}   (Spearman with sensitivity {rho:+.2f})")

        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.barh(imp.driver[::-1], imp.importance[::-1], xerr=imp.sd[::-1], color="#1f78b4")
        ax.set_xlabel("Permutation importance on held-out blocks (drop in R2)")
        ax.set_title("Drivers of drought sensitivity")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig11_driver_importance.png")

        top = imp.driver.head(4).tolist()
        fig, axes = plt.subplots(1, len(top), figsize=(3 * len(top), 3), sharey=True)
        PartialDependenceDisplay.from_estimator(rf, X, top, ax=np.atleast_1d(axes))
        fig.suptitle("How each driver changes drought sensitivity (partial dependence)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig12_partial_dependence.png")
    else:
        say("")
        say("D skipped: gaya_pixel_sensitivity.csv has no driver columns yet - re-run the updated Script 2.")

# =====================================================================
# E. ROBUSTNESS OF THE BUFFERING RESULT
# =====================================================================
if exists("vhi_merged.csv", folder=OUT3_DIR):
    say("")
    say("--- E. ROBUSTNESS: drop in VHI (normal minus drought years), irrigated vs rainfed ---")
    v = pd.read_csv(OUT3_DIR / "vhi_merged.csv")
    v = v[v.group.isin(COLORS)]
    variants = [("Detrended VHI, SPEI-4 <= -1.0 (main)", "VHI", "SPEI", -1.0),
                ("Raw VHI, SPEI-4 <= -1.0", "VHI_raw", "SPEI", -1.0),
                ("Detrended VHI, SPEI-4 <= -0.8", "VHI", "SPEI", -0.8),
                ("Detrended VHI, SPI-4 <= -1.0", "VHI", "SPI", -1.0)]
    rob = []
    for season in ["kharif", "rabi"]:
        for name, col, idx, thr in variants:
            d = v[(v.season == season)].dropna(subset=[col, idx])
            pv = d.pivot_table(index="year", columns="group", values=col)
            dry = d.drop_duplicates("year").set_index("year")[idx].reindex(pv.index) <= thr
            if dry.sum() < 2:
                continue
            drop = pv[~dry].mean() - pv[dry].mean()
            gap = pv.Irrigated - pv.Rainfed
            u = stats.mannwhitneyu(gap[dry], gap[~dry], alternative="greater")
            rob.append(dict(season=season, variant=name, n_drought=int(dry.sum()),
                            drop_irrigated=drop.Irrigated, drop_rainfed=drop.Rainfed,
                            buffering_points=drop.Rainfed - drop.Irrigated, p_gap_widens=u.pvalue))
            say(f"{season:6s} {name:38s} n={int(dry.sum())}: irrigated {drop.Irrigated:5.1f}, rainfed "
                f"{drop.Rainfed:5.1f}, buffering {drop.Rainfed - drop.Irrigated:+5.1f} points (p={u.pvalue:.2f})")
        # leave one drought year out (main variant)
        d = v[v.season == season]
        pv = d.pivot_table(index="year", columns="group", values="VHI")
        dry = d.drop_duplicates("year").set_index("year").SPEI.reindex(pv.index) <= DROUGHT_T
        outs = []
        for yr in pv.index[dry]:
            keep = pv.index != yr
            dd = pv[keep & ~dry].mean() - pv[keep & dry].mean()
            outs.append(dd.Rainfed - dd.Irrigated)
        if outs:
            say(f"{season:6s} leave-one-drought-year-out buffering: {min(outs):+.1f} to {max(outs):+.1f} points")
            rob.append(dict(season=season, variant="leave one drought year out (range)",
                            n_drought=int(dry.sum()) - 1, buffering_points=np.nan,
                            drop_irrigated=min(outs), drop_rainfed=max(outs), p_gap_widens=np.nan))
    pd.DataFrame(rob).round(3).to_csv(OUT_DIR / "E_robustness.csv", index=False)

(OUT_DIR / "novelty_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print(f"\nDone. Results in {OUT_DIR}")
