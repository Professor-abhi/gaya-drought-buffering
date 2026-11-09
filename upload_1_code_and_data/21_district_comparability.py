"""
SCRIPT 21 - Reviewer concern M7: are the Sone canal districts (Rohtas, Bhojpur) comparable to the monsoon-fed ones?

For each of the seven districts (from Script 10 per-district outputs and region_climate_monthly.csv):
  mean and CV of June-September rainfall, mean Aug-Oct Tmax, number of drought years 2001-2025, mean SPEI and
  rainfall in those drought years, and the Kharif VHI drop of the rainfed group (how hard drought hit crops).
Run:    python 21_district_comparability.py   ->  outputs_region/district_comparability.csv / .txt
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REG = HERE / "outputs_region"
CANAL = {"Rohtas", "Bhojpur"}
cm = pd.read_csv(HERE / "data" / "region_climate_monthly.csv")
cm["yr"], cm["mo"] = cm.date.str[:4].astype(int), cm.date.str[5:7].astype(int)
summ = pd.read_csv(REG / "regional_summary.csv")
rows = []
for dist in sorted(cm.district.unique()):
    c = cm[cm.district == dist]
    jjas = c[c.mo.between(6, 9)].groupby("yr").pr_mm.sum()
    tko = c[(c.mo.between(8, 10)) & (c.yr.between(2001, 2025))].groupby("yr").tmax_C.mean()
    a = pd.read_csv(REG / dist / "out3" / "annual_drought_indices.csv").set_index("year")
    a = a[(a.index >= 2001) & (a.index <= 2025)]
    dry = a[a.SPEI <= -1.0]
    k = summ[(summ.district == dist) & (summ.season == "kharif")].iloc[0]
    rows.append(dict(district=dist, water_source="Sone canal" if dist in CANAL else "monsoon-fed",
                     JJAS_mean_mm=jjas.loc[1981:2025].mean(), JJAS_CV_pct=100 * jjas.loc[1981:2025].std() / jjas.loc[1981:2025].mean(),
                     AugOct_Tmax_C=tko.mean(), n_drought_2001_2025=len(dry),
                     drought_years=" ".join(str(v) for v in dry.index),
                     mean_SPEI_in_droughts=dry.SPEI.mean(), mean_JJAS_in_droughts_mm=dry.JJAS_rain_mm.mean(),
                     JJAS_deficit_in_droughts_pct=100 * (dry.JJAS_rain_mm.mean() / jjas.loc[1981:2025].mean() - 1),
                     kharif_VHI_drop_rainfed=k.drop_rainfed, kharif_VHI_drop_irrigated=k.drop_irrigated))
t = pd.DataFrame(rows).sort_values(["water_source", "district"])
t.round(2).to_csv(REG / "district_comparability.csv", index=False)
txt = t.round(2).to_string(index=False)
grp = t.groupby("water_source")[["JJAS_mean_mm", "JJAS_CV_pct", "mean_SPEI_in_droughts",
                                  "JJAS_deficit_in_droughts_pct", "kharif_VHI_drop_rainfed"]].mean().round(2)
(REG / "district_comparability.txt").write_text(txt + "\n\nGroup means\n" + grp.to_string(), encoding="utf-8")
print(txt)
print()
print(grp.to_string())
