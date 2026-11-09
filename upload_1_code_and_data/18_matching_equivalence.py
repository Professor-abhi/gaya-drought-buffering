"""
SCRIPT 18 - Reviewer concerns M2 and M3

  A. Matched comparison (M3): irrigated (grp 1) vs rainfed (grp 2) 250 m pixels matched on terrain, soil and
     climate (propensity score, 1:1 nearest neighbour with replacement, caliper 0.2 SD of the logit). Outcome:
     per-pixel Kharif drought sensitivity (VHI points per SD of monsoon rainfall). CIs by spatial-block
     bootstrap (0.1 deg blocks) to respect spatial autocorrelation.
  B. Equivalence / detectable effect (M2): year-level buffering (rainfed drop minus irrigated drop, detrended VHI,
     drought = SPEI-4 <= -1) for the main and the persistent groups. One-sided 95% upper bound (TOST logic)
     expressed as % of the rainfed loss, and the minimum detectable buffering at 80% power.

Inputs: data/gaya_pixel_sensitivity.csv, data/gaya_vhi_by_group.csv, data/gaya_vhi_revision_groups.csv,
        outputs/annual_drought_indices.csv
Run:    python 18_matching_equivalence.py   ->  outputs_novelty/K_matching_equivalence.txt / .csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "outputs_novelty"
rng = np.random.default_rng(18)
N_BOOT = 4000
COV = ["elev_m", "slope_deg", "dist_river_m", "clay_pct", "rain_cv", "mar_tmax_C"]
lines, rows = [], []


def say(t=""):
    print(t)
    lines.append(t)


# ---------------- A. matched pixel comparison ----------------
px = pd.read_csv(DATA / "gaya_pixel_sensitivity.csv")
px = px[px.grp.isin([1, 2])].dropna(subset=COV + ["slope"]).reset_index(drop=True)
xy = np.array([json.loads(g)["coordinates"] for g in px[".geo"]])
px["block"] = (np.floor(xy[:, 0] / 0.1).astype(int) * 1000 + np.floor(xy[:, 1] / 0.1).astype(int))
px["T"] = (px.grp == 1).astype(int)
Z = (px[COV] - px[COV].mean()) / px[COV].std()
ps = LogisticRegression(max_iter=2000).fit(Z, px["T"]).predict_proba(Z)[:, 1]
px["logit"] = np.log(ps / (1 - ps))


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def match(d):
    """1:1 nearest-neighbour on the logit, with replacement, caliper 0.2 SD -> (treated idx, control idx)."""
    t, c = d[d["T"] == 1], d[d["T"] == 0]
    if len(t) == 0 or len(c) == 0:
        return None
    nn = NearestNeighbors(n_neighbors=1).fit(c[["logit"]])
    dist, j = nn.kneighbors(t[["logit"]])
    ok = dist[:, 0] <= 0.2 * d["logit"].std()
    return t.index[ok], c.index[j[ok, 0]]


ti, ci = match(px)
say("A. MATCHED COMPARISON OF PIXEL DROUGHT SENSITIVITY (Kharif VHI points per SD of monsoon rainfall)")
say(f"Pixels: {px['T'].sum()} irrigated, {(1 - px['T']).sum()} rainfed; matched irrigated pixels {len(ti)} "
    f"({len(set(ci))} distinct rainfed controls)")
say("Covariate balance, standardized mean difference (|SMD| < 0.1 = balanced):")
for v in COV:
    say(f"  {v:13s} before {smd(px.loc[px['T'] == 1, v], px.loc[px['T'] == 0, v]):+.2f}   "
        f"after {smd(px.loc[ti, v], px.loc[ci, v]):+.2f}")
raw = px.loc[px["T"] == 1, "slope"].mean() - px.loc[px["T"] == 0, "slope"].mean()
att = (px.loc[ti, "slope"].values - px.loc[ci, "slope"].values).mean()
blocks = px.block.unique()
boot_raw, boot_att = [], []
for _ in range(N_BOOT):
    pick = rng.choice(blocks, len(blocks))
    d = pd.concat([px[px.block == b] for b in pick], ignore_index=True)
    m = match(d)
    if m is None or len(m[0]) == 0:
        continue
    boot_raw.append(d.loc[d["T"] == 1, "slope"].mean() - d.loc[d["T"] == 0, "slope"].mean())
    boot_att.append((d.loc[m[0], "slope"].values - d.loc[m[1], "slope"].values).mean())
rlo, rhi = np.percentile(boot_raw, [2.5, 97.5])
alo, ahi = np.percentile(boot_att, [2.5, 97.5])
rain_mean = px.loc[px["T"] == 0, "slope"].mean()
say(f"Mean sensitivity, rainfed pixels: {rain_mean:.2f}")
say(f"Irrigated minus rainfed, unmatched: {raw:+.2f} (spatial-block 95% CI {rlo:+.2f} to {rhi:+.2f})")
say(f"Irrigated minus rainfed, matched:   {att:+.2f} (spatial-block 95% CI {alo:+.2f} to {ahi:+.2f}); "
    f"i.e. {100 * att / rain_mean:+.0f}% of the rainfed sensitivity (CI {100 * alo / rain_mean:+.0f}% to "
    f"{100 * ahi / rain_mean:+.0f}%)")
say("(negative = irrigated pixels less drought-sensitive, i.e. buffering)")
rows.append(dict(test="pixel_matched", estimate=att, lo=alo, hi=ahi, reference=rain_mean))
rows.append(dict(test="pixel_unmatched", estimate=raw, lo=rlo, hi=rhi, reference=rain_mean))

# ---------------- B. equivalence and minimum detectable effect ----------------
a = pd.read_csv(HERE / "outputs" / "annual_drought_indices.csv").set_index("year")


def buffering_test(path, g_irr, g_ref, label):
    v = pd.read_csv(path)
    d = v[(v.season == "kharif") & v.grp.isin([g_irr, g_ref])].pivot_table(index="year", columns="grp", values="VHI")
    d = d.dropna()
    yrs = d.index.values
    X = d[[g_irr, g_ref]].to_numpy(float).copy()
    for j in range(2):
        X[:, j] -= np.polyval(np.polyfit(yrs, X[:, j], 1), yrs)
    dry = (a.SPEI.reindex(yrs) <= -1.0).to_numpy()

    def stat(ix):
        x, m = X[ix], dry[ix]
        if m.sum() == 0 or (~m).sum() == 0:
            return np.nan, np.nan
        drop = x[~m].mean(0) - x[m].mean(0)
        return drop[1] - drop[0], drop[1]          # buffering, reference (rainfed) drop

    est, ref = stat(np.arange(len(yrs)))
    boot = np.array([stat(rng.integers(0, len(yrs), len(yrs))) for _ in range(N_BOOT)])
    b = boot[~np.isnan(boot[:, 0]), 0]
    up95 = np.percentile(b, 95)                    # one-sided 95% upper bound (TOST at alpha = 0.05)
    se = b.std(ddof=1)
    mde = (norm.ppf(0.975) + norm.ppf(0.80)) * se  # two-sided alpha 0.05, power 0.80
    say(f"{label}: {dry.sum()} drought years of {len(yrs)}; rainfed loss {ref:.1f} VHI points; "
        f"buffering {est:+.1f}; one-sided 95% upper bound {up95:+.1f} = {100 * up95 / ref:.0f}% of the loss; "
        f"bootstrap SE {se:.1f}; minimum detectable buffering (80% power) {mde:.1f} points = {100 * mde / ref:.0f}% "
        f"of the loss")
    rows.append(dict(test=label, estimate=est, lo=np.nan, hi=up95, reference=ref, se=se, mde=mde))
    return up95 / ref


say("")
say("B. EQUIVALENCE (TOST-style) AND MINIMUM DETECTABLE EFFECT, year-level Kharif buffering, detrended VHI")
f1 = buffering_test(DATA / "gaya_vhi_by_group.csv", 1, 2, "Main groups (irrigated vs rainfed)")
f2 = buffering_test(DATA / "gaya_vhi_revision_groups.csv", 1, 2, "Persistent groups (persistently irrigated vs rainfed)")
for margin in (0.25, 0.33, 0.50):
    say(f"Buffering of >= {100 * margin:.0f}% of the rainfed loss rejected at alpha = 0.05 (one-sided): "
        f"main {'yes' if f1 < margin else 'no'}, persistent {'yes' if f2 < margin else 'no'}")

pd.DataFrame(rows).round(3).to_csv(OUT / "K_matching_equivalence.csv", index=False)
(OUT / "K_matching_equivalence.txt").write_text("\n".join(lines), encoding="utf-8")
