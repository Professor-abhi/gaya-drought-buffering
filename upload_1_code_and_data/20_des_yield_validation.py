"""
SCRIPT 20 - Yield validation with DES district statistics (reviewer concern M5 and the 2009-10 vs 2022 contrast)

Data: DES Area, Production & Yield, Gaya, rice (all seasons summed) and wheat (Rabi), agricultural years
      1997-98 to 2022-23 (data.desagri.gov.in, Screen View, accessed 2026-11-02)
      -> data/des_gaya_rice_wheat_apy_1997_2022.csv
  1. Data checks: incomplete years; implausible yields; agreement with the ICRISAT series (1997-2015 overlap)
  2. Rice: % anomaly from a Theil-Sen trend in each drought year (SPEI-4 <= -1: 2009, 2010, 2022) and per SPEI unit
  3. Wheat: % anomaly vs standardized Feb-Mar Tmax of the harvest year (and the preceding monsoon SPEI);
     correlation with detrended Rabi VHI of the irrigated and rainfed groups (2001-2022)
Run:    python 20_des_yield_validation.py   ->  outputs_yield/DES_yield_validation.txt / .csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "outputs_yield"
OUT.mkdir(exist_ok=True)
lines = []


def say(t=""):
    print(t)
    lines.append(t)


d = pd.read_csv(DATA / "des_gaya_rice_wheat_apy_1997_2022.csv")
d["year"] = d.Year.str[:4].astype(int)                          # agricultural year y = Kharif y, Rabi harvest y+1
ra = [c for c in d.columns if c.startswith("Rice_") and c.endswith("_Area")]
rp = [c for c in d.columns if c.startswith("Rice_") and c.endswith("_Production")]
y = pd.DataFrame({"year": d.year, "rice_area": d[ra].sum(1, min_count=1), "rice_prod": d[rp].sum(1, min_count=1),
                  "wheat_area": d.Wheat_Rabi_Area, "wheat_prod": d.Wheat_Rabi_Production}).set_index("year")
y["rice_yield"] = y.rice_prod / y.rice_area
y["wheat_yield"] = y.wheat_prod / y.wheat_area

a = pd.read_csv(HERE / "outputs" / "annual_drought_indices.csv").set_index("year")
cm = pd.read_csv(DATA / "gaya_climate_monthly.csv")
cm["yr"], cm["mo"] = cm.date.str[:4].astype(int), cm.date.str[5:7].astype(int)
heat = cm[cm.mo.isin([2, 3])].groupby("yr").tmax_C.mean().rename(lambda v: v - 1)   # Feb-Mar of y+1 -> year y

say("DES YIELD VALIDATION, GAYA (rice = all seasons; wheat = Rabi), agricultural years 1997-2022")
# ---- 1. data checks
med_area = y.rice_area.median()
incomplete = y.index[y.rice_area < 0.1 * med_area].tolist()
low = y.index[y.rice_yield < 0.3 * y.rice_yield.median()].tolist()
say(f"Rice: incomplete years (area < 10% of median) {incomplete}; implausibly low yield (< 30% of median) {low}")
ic = pd.read_csv(DATA / "icrisat_gaya_apy_mendeley.csv")
ic = ic.set_index("Year")
icy = ic["RICE PRODUCTION (1000 tons)"] * 1000 / (ic["RICE AREA (1000 ha)"] * 1000)
for yr in low:
    if yr in icy.index:
        say(f"  {yr}: DES rice yield {y.rice_yield[yr]:.2f} t/ha; ICRISAT {icy[yr]:.2f} t/ha")
ov = pd.concat([y.rice_prod, ic["RICE PRODUCTION (1000 tons)"] * 1000], axis=1, keys=["des", "icr"]).dropna()
ov = ov.drop([v for v in incomplete + low if v in ov.index])
say(f"DES vs ICRISAT rice production, {ov.index.min()}-{ov.index.max()} ({len(ov)} years, flagged years removed): "
    f"r = {np.corrcoef(ov.des, ov.icr)[0, 1]:.2f}")


def anomaly(s):
    """% departure from a Theil-Sen trend."""
    s = s.dropna()
    sl, ic0, _, _ = stats.theilslopes(s.values, s.index.values)
    tr = ic0 + sl * s.index.values
    return pd.Series(100 * (s.values - tr) / tr, index=s.index)


def ols(x, yv):
    r = stats.linregress(x, yv)
    return r.slope, r.pvalue


rows = []
for label, drop in [("main (incomplete years removed)", incomplete),
                    ("sensitivity (also implausible low-yield years removed)", incomplete + low)]:
    say("")
    say(f"--- RICE, {label}")
    r = y.drop([v for v in drop if v in y.index])
    for var in ["rice_area", "rice_prod", "rice_yield"]:
        an = anomaly(r[var])
        sp = a.SPEI.reindex(an.index)
        dry = an[sp <= -1.0]
        slope, p = ols(sp.values, an.values)
        say(f"{var:10s} drought-year anomalies: " + ", ".join(f"{k} {v:+.0f}%" for k, v in dry.items())
            + f" | other years mean {an[sp > -1.0].mean():+.0f}% | per SPEI unit {slope:+.1f}% (p = {p:.3f}), n = {len(an)}")
        rows += [dict(analysis=label, variable=var, year=k, anomaly_pct=v) for k, v in dry.items()]

# ---- 3. wheat and heat
say("")
say("--- WHEAT (Rabi), heat = standardized Feb-Mar mean Tmax of the harvest year (ERA5-Land)")
w = y.drop([v for v in incomplete if v in y.index and False])          # wheat series is complete
for var in ["wheat_yield", "wheat_prod"]:
    an = anomaly(w[var])
    h = heat.reindex(an.index)
    hz = (h - h.mean()) / h.std()
    slope, p = ols(hz.values, an.values)
    X = np.column_stack([np.ones(len(an)), hz.values, a.SPEI.reindex(an.index).values])
    b, *_ = np.linalg.lstsq(X, an.values, rcond=None)
    res = an.values - X @ b
    se = np.sqrt(np.diag(res @ res / (len(an) - 3) * np.linalg.inv(X.T @ X)))
    pv = 2 * stats.t.sf(np.abs(b / se), len(an) - 3)
    say(f"{var:11s}: {slope:+.1f}% per SD of heat (p = {p:.3f}, n = {len(an)}); with monsoon SPEI: heat {b[1]:+.1f}% "
        f"(p = {pv[1]:.3f}), SPEI {b[2]:+.1f}% (p = {pv[2]:.3f})")
    hot = an[hz >= 1.0]
    say(f"             hot years (>= +1 SD): " + ", ".join(f"{k} {v:+.0f}%" for k, v in hot.items()))
    rows += [dict(analysis="wheat", variable=var, year=k, anomaly_pct=v) for k, v in an.items()]

# rainfed (residual-moisture) Rabi crops: a heat-exposed comparison for the mostly irrigated wheat
say("")
say("--- RABI PULSES AND MUSTARD (mostly rainfed / residual moisture), same heat index")
pm = pd.read_csv(DATA / "des_gaya_rabi_pulses_mustard_1997_2022.csv")
pm["year"] = pm.Year.str[:4].astype(int)
pm = pm.set_index("year")
crops = {"wheat (mostly irrigated)": w.wheat_yield, "gram": pm.Gram_Rabi_Production / pm.Gram_Rabi_Area,
         "masoor (lentil)": pm.Masoor_Rabi_Production / pm.Masoor_Rabi_Area,
         "rapeseed-mustard": pm.RapeseedMustard_Rabi_Production / pm.RapeseedMustard_Rabi_Area}
pulses = pd.concat([crops["gram"], crops["masoor (lentil)"]], axis=1).mean(1, skipna=False)
crops["gram + masoor (mean yield anomaly)"] = None
an_all = {}
for name, s in crops.items():
    if s is None:
        an = pd.concat([an_all["gram"], an_all["masoor (lentil)"]], axis=1).dropna().mean(1)
    else:
        an = anomaly(s)
        an_all[name] = an
    h = heat.reindex(an.index)
    hz = (h - h.mean()) / h.std()
    slope, p = ols(hz.values, an.values)
    say(f"{name:36s}: {slope:+.1f}% yield per SD of Feb-Mar heat (p = {p:.3f}, n = {len(an)})")
    rows += [dict(analysis="rabi_heat", variable=name, year=k, anomaly_pct=v) for k, v in an.items()]
gap = (an_all["wheat (mostly irrigated)"] - pd.concat([an_all["gram"], an_all["masoor (lentil)"]], axis=1).mean(1)).dropna()
hz = (heat.reindex(gap.index) - heat.reindex(gap.index).mean()) / heat.reindex(gap.index).std()
slope, p = ols(hz.values, gap.values)
say(f"Wheat minus pulse yield anomaly vs heat: {slope:+.1f} percentage points per SD (p = {p:.3f}, n = {len(gap)}) "
    f"(positive = irrigated wheat holds up better than rainfed pulses in hot years)")

v = pd.read_csv(DATA / "gaya_vhi_by_group.csv")
wa = anomaly(w.wheat_yield)
for g, name in [(1, "irrigated"), (2, "rainfed")]:
    s = v[(v.season == "rabi") & (v.grp == g)].set_index("year").VHI
    s = s - np.polyval(np.polyfit(s.index, s.values, 1), s.index)
    j = pd.concat([s, wa], axis=1, keys=["vhi", "wheat"]).dropna()
    rr, pp = stats.pearsonr(j.vhi, j.wheat)
    say(f"Detrended Rabi VHI ({name} pixels) vs wheat yield anomaly, {j.index.min()}-{j.index.max()}: r = {rr:.2f} (p = {pp:.3f})")
ka = anomaly(y.drop(incomplete).rice_prod)
for g, name in [(1, "irrigated"), (2, "rainfed")]:
    s = v[(v.season == "kharif") & (v.grp == g)].set_index("year").VHI
    s = s - np.polyval(np.polyfit(s.index, s.values, 1), s.index)
    j = pd.concat([s, ka], axis=1, keys=["vhi", "rice"]).dropna()
    rr, pp = stats.pearsonr(j.vhi, j.rice)
    say(f"Detrended Kharif VHI ({name} pixels) vs rice production anomaly, {j.index.min()}-{j.index.max()}: "
        f"r = {rr:.2f} (p = {pp:.3f})")

pd.DataFrame(rows).round(2).to_csv(OUT / "DES_yield_validation.csv", index=False)
(OUT / "DES_yield_validation.txt").write_text("\n".join(lines), encoding="utf-8")
