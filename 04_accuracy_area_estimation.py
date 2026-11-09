"""
SCRIPT 4 - Accuracy assessment and error-adjusted area estimation
(Olofsson et al., 2014, Remote Sensing of Environment 148: 42-57)

Uses the blind validation points (stratified random sample by map class) and the mapped
area of each class to give:
  - overall, user's and producer's accuracy with 95% confidence intervals
  - error-adjusted (unbiased) area of each class with 95% confidence interval
  - 2-class version: Irrigated vs Not irrigated

Inputs (in ./data):
    gaya_validation_points_2023.csv   (Script 1b, "Finish & compute accuracy" export)  map, ref
    gaya_irrigated_area_by_year.csv   (Script 1b, validate mode)  year, ha_class0..2
If the validation CSV is missing, the confusion matrix typed below is used.

Run:
    python 04_accuracy_area_estimation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
YEAR = 2023
NAMES = ["Irrigated", "Rainfed", "Fallow"]

# Fallback: rows = YOUR label (reference), columns = MAP (as printed by GEE), 163 points
CM_REF_BY_MAP = [[52, 15, 1],
                 [3, 38, 13],
                 [0, 2, 39]]
# Fallback mapped areas (ha) - REPLACE with the 2023 RF values if the area CSV is missing
MAPPED_HA = [263131.7, 96240.8, 2875.5]   # RF 2023 map, GEE reduceRegion at 30 m (2026-10-24)


def load_counts():
    f = DATA_DIR / f"gaya_validation_points_{YEAR}.csv"
    if f.exists():
        v = pd.read_csv(f)
        v = v[v.ref >= 0]
        n = pd.crosstab(v["map"], v["ref"]).reindex(index=range(3), columns=range(3), fill_value=0)
        print(f"Validation points read from {f.name}: {len(v)}")
        return n.to_numpy(float)
    print("Validation CSV not found - using the confusion matrix typed in the script.")
    return np.array(CM_REF_BY_MAP, float).T          # -> rows = map, columns = reference


def load_areas():
    f = DATA_DIR / "gaya_irrigated_area_by_year.csv"
    if f.exists():
        a = pd.read_csv(f)
        a = a[a.year == YEAR]
        if len(a):
            r = a.iloc[-1]
            return np.array([r.ha_class0, r.ha_class1, r.ha_class2], float)
    if MAPPED_HA is not None:
        return np.array(MAPPED_HA, float)
    raise SystemExit("Mapped areas not found: put gaya_irrigated_area_by_year.csv in data/ "
                     "or type the 2023 hectares into MAPPED_HA.")


def olofsson(n, A, names):
    """n: counts, rows = map class i, cols = reference class j. A: mapped area per map class."""
    k = len(A)
    n_i = n.sum(1)
    W = A / A.sum()
    p = W[:, None] * n / n_i[:, None]                  # estimated area proportions
    p_j = p.sum(0)
    oa = np.trace(p)
    ua = np.diag(n) / n_i
    pa = np.diag(p) / p_j
    se_oa = np.sqrt(np.sum(W ** 2 * ua * (1 - ua) / (n_i - 1)))
    se_ua = np.sqrt(ua * (1 - ua) / (n_i - 1))
    se_pj = np.sqrt([np.sum(W ** 2 * (n[:, j] / n_i) * (1 - n[:, j] / n_i) / (n_i - 1)) for j in range(k)])
    # producer's accuracy SE (Olofsson et al. 2014, eq. 7), in area units
    N_i = A
    N_j = np.array([np.sum(N_i / n_i * n[:, j]) for j in range(k)])
    se_pa = []
    for j in range(k):
        t1 = N_i[j] ** 2 * (1 - pa[j]) ** 2 * ua[j] * (1 - ua[j]) / (n_i[j] - 1)
        t2 = pa[j] ** 2 * sum(N_i[i] ** 2 * (n[i, j] / n_i[i]) * (1 - n[i, j] / n_i[i]) / (n_i[i] - 1)
                              for i in range(k) if i != j)
        se_pa.append(np.sqrt(t1 + t2) / N_j[j])
    se_pa = np.array(se_pa)
    rows = []
    for j in range(k):
        rows.append(dict(
            cls=names[j], n_samples=int(n_i[j]), mapped_ha=round(A[j]),
            users_acc=round(ua[j] * 100, 1), users_ci95=round(1.96 * se_ua[j] * 100, 1),
            producers_acc=round(pa[j] * 100, 1), producers_ci95=round(1.96 * se_pa[j] * 100, 1),
            adjusted_ha=round(p_j[j] * A.sum()), adjusted_ci95_ha=round(1.96 * se_pj[j] * A.sum())))
    return pd.DataFrame(rows), oa, 1.96 * se_oa


n = load_counts()
A = load_areas()
lines = []

tab3, oa3, ci3 = olofsson(n, A, NAMES)
lines.append(f"THREE CLASSES ({YEAR}) - overall accuracy {oa3 * 100:.1f}% +/- {ci3 * 100:.1f}% (95% CI)")
lines.append(tab3.to_string(index=False))

# 2 classes: Irrigated vs Not irrigated (rainfed + fallow)
n2 = np.array([[n[0, 0], n[0, 1:].sum()],
               [n[1:, 0].sum(), n[1:, 1:].sum()]])
A2 = np.array([A[0], A[1:].sum()])
tab2, oa2, ci2 = olofsson(n2, A2, ["Irrigated", "Not irrigated"])
lines.append("")
lines.append(f"TWO CLASSES ({YEAR}) - overall accuracy {oa2 * 100:.1f}% +/- {ci2 * 100:.1f}% (95% CI)")
lines.append(tab2.to_string(index=False))

irr = tab3.iloc[0]
lines.append("")
lines.append(f"Irrigated area {YEAR}: mapped {irr.mapped_ha:,.0f} ha; error-adjusted "
             f"{irr.adjusted_ha:,.0f} +/- {irr.adjusted_ci95_ha:,.0f} ha (95% CI)")

text = "\n".join(lines)
print(text)
(OUT_DIR / f"accuracy_area_{YEAR}.txt").write_text(text, encoding="utf-8")
tab3.to_csv(OUT_DIR / f"accuracy_area_3class_{YEAR}.csv", index=False)
tab2.to_csv(OUT_DIR / f"accuracy_area_2class_{YEAR}.csv", index=False)
