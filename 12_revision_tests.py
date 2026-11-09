"""
SCRIPT 12 - Reviewer concerns 2 and 5: tests with the revision groups from Script 2c

  Kharif drought test : 1 persistently irrigated  vs  2 persistently rainfed (Kharif-only)
  Rabi heat test      : 1 persistently irrigated  vs  3 rainfed Rabi crops (crop vs crop)
                        (and 1 vs 2 for comparison with the original design)
Each test: detrended VHI (and VCI), drop in drought years with bootstrap 95% CI, and the paired
year-matched regression gap = a + c1*SPEI + c2*heat with bootstrap CIs.

Inputs: data/gaya_vhi_revision_groups.csv, data/gaya_revision_group_areas.csv (optional),
        data/gaya_climate_monthly.csv, outputs/annual_drought_indices.csv
Run:    python 12_revision_tests.py   ->  outputs_novelty/J_revision_tests.txt / .csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT3 = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "outputs"
OUT = HERE / "outputs_novelty"
NAMES = {1: "persistently irrigated", 2: "persistently rainfed (Kharif-only)", 3: "rainfed Rabi crops"}
rng = np.random.default_rng(7)
lines, rows = [], []


def say(t=""):
    print(t)
    lines.append(t)


v = pd.read_csv(DATA / "gaya_vhi_revision_groups.csv")
a = pd.read_csv(OUT3 / "annual_drought_indices.csv").set_index("year")
cm = pd.read_csv(DATA / "gaya_climate_monthly.csv")
cm["yr"], cm["mo"] = cm.date.str[:4].astype(int), cm.date.str[5:7].astype(int)
heat = {"kharif": cm[cm.mo.isin([8, 9, 10])].groupby("yr").tmax_C.mean(),
        "rabi": cm[cm.mo.isin([2, 3])].groupby("yr").tmax_C.mean().rename(lambda y: y - 1)}
say("REVISION TESTS (reviewer concerns 2 and 5) - Gaya")
af = DATA / "gaya_revision_group_areas.csv"
if af.exists():
    ar = pd.read_csv(af).set_index("grp").area_ha
    say("Group areas (ha): " + ", ".join(f"{NAMES.get(int(g), g)} {ar[g]:,.0f}" for g in ar.index))


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def test(season, g_irr, g_ref, var):
    d = v[(v.season == season) & v.grp.isin([g_irr, g_ref])].pivot_table(index="year", columns="grp", values=var).dropna()
    if d.shape[1] < 2 or len(d) < 10:
        return None
    yrs = d.index.values
    X = d[[g_irr, g_ref]].to_numpy(float).copy()
    for j in range(2):
        X[:, j] -= np.polyval(np.polyfit(yrs, X[:, j], 1), yrs)
    spei = a.SPEI.reindex(yrs).to_numpy()
    h = heat[season].reindex(yrs)
    hz = ((h - heat[season].mean()) / heat[season].std()).to_numpy()
    dry = spei <= -1.0

    def stats_(ix):
        x, m = X[ix], dry[ix]
        buf = np.nan
        if m.sum() and (~m).sum():
            dd = x[~m].mean(0) - x[m].mean(0)
            buf = dd[1] - dd[0]                      # reference drop minus irrigated drop
        gap = x[:, 0] - x[:, 1]
        b = ols(np.column_stack([spei[ix], hz[ix]]), gap)
        return buf, b[1], b[2]

    est = stats_(np.arange(len(yrs)))
    boot = np.array([stats_(rng.integers(0, len(yrs), len(yrs))) for _ in range(3000)])
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    ref_drop = (X[~dry].mean(0) - X[dry].mean(0))[1] if dry.sum() else np.nan
    p_heat = 2 * min(np.mean(boot[:, 2] <= 0), np.mean(boot[:, 2] >= 0))
    return dict(season=season, comparison=f"{NAMES[g_irr]} vs {NAMES[g_ref]}", index=var, n_years=len(yrs),
                n_drought=int(dry.sum()), reference_drop=ref_drop, buffering=est[0], buf_lo=lo[0], buf_hi=hi[0],
                gap_per_SPEI=est[1], spei_lo=lo[1], spei_hi=hi[1], gap_per_heat=est[2], heat_lo=lo[2], heat_hi=hi[2],
                p_heat_boot=p_heat)


for season, pairs in [("kharif", [(1, 2)]), ("rabi", [(1, 3), (1, 2)])]:
    for gi, gr in pairs:
        for var in ["VHI", "VCI"]:
            r = test(season, gi, gr, var)
            if r is None:
                say(f"{season}: {NAMES[gi]} vs {NAMES[gr]} - not enough data")
                continue
            rows.append(r)
            say(f"{season:6s} {var} {r['comparison']}: buffering {r['buffering']:+.1f} "
                f"[{r['buf_lo']:+.1f}, {r['buf_hi']:+.1f}] of a {r['reference_drop']:.1f}-point loss; "
                f"gap per SD heat {r['gap_per_heat']:+.2f} [{r['heat_lo']:+.2f}, {r['heat_hi']:+.2f}] "
                f"(bootstrap p {r['p_heat_boot']:.3f}); gap per SPEI {r['gap_per_SPEI']:+.2f} "
                f"[{r['spei_lo']:+.2f}, {r['spei_hi']:+.2f}]")
pd.DataFrame(rows).round(3).to_csv(OUT / "J_revision_tests.csv", index=False)
(OUT / "J_revision_tests.txt").write_text("\n".join(lines), encoding="utf-8")
