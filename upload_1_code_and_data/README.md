# Monsoon drought and terminal heat buffering by winter irrigation in South Bihar, India

Code and derived data for the paper **"Monsoon drought and terminal heat buffering by winter irrigation in South Bihar, India: a multi-sensor satellite assessment"** (manuscript submitted for publication).

The study asks whether land irrigated in the winter (Rabi) season is protected against monsoon (Kharif) drought and February–March heat in Gaya district, Bihar. It combines several sources:
- a Sentinel-2/Sentinel-1 Random Forest map of Rabi-cropped land;
- ESA WorldCereal;
- MODIS Vegetation Health Index (2001–2025);
- SPEI from CHIRPS and ERA5-Land;
- GRACE/GRACE-FO groundwater storage;
- CGWB observation wells;
- official crop statistics.

Six neighbouring districts serve as a replication.

## Repository layout
| Path | Contents |
|---|---|
| `*.js` | Google Earth Engine scripts (run in the Earth Engine Code Editor) |
| `*.py` | Python analysis and figure scripts |
| `data/` | Tables exported from Earth Engine, derived crop statistics and validation labels |
| `outputs/`, `outputs_novelty/`, `outputs_yield/`, `outputs_region/` | Results and figures produced by the Python scripts |

## Requirements
- **Google Earth Engine:** an account and a Cloud project. In the `.js` scripts, set `ASSET_ROOT` and `S2_ASSET` to your own project.
- **Python 3.10+:** `pip install -r requirements.txt`

## Order of use
| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `01_gaya_irrigated_area_mapping.js` | Rule-based first map (Otsu threshold) and NDVI inspection | map preview |
| 1 | `01b_gaya_auto_training_RF.js` | Automatic crop-calendar labels, Random Forest map (`MODE` = export or validate) and blind validation panel | `gaya_irrigation_RF_2023` asset, `gaya_irrigated_area_by_year.csv`, `gaya_validation_points_2023.csv` |
| 1 | `01c_gaya_irrigation_v2.js` | Stricter v2 sensitivity map | preview |
| 1 | `01e_second_interpreter.js` | Blind 50-point relabelling tool | `repeat_labels.txt` |
| 2 | `02_gaya_drought_indices.js` | VCI, TCI and VHI by group, climate series, pixel sensitivity and drivers | `gaya_vhi_by_group.csv`, `gaya_climate_monthly.csv`, `gaya_pixel_sensitivity.csv` |
| 2 | `02b_south_bihar_region.js` | The same for six neighbouring districts | `region_vhi_by_group.csv`, `region_climate_monthly.csv` |
| 2 | `02c_revision_groups.js` | Persistently irrigated, persistently rainfed and rainfed-Rabi-crop groups | `gaya_vhi_revision_groups.csv`, `gaya_revision_group_areas.csv` |
| 3 | `03_drought_buffering_analysis.py` | SPI/SPEI, detrending and buffering tests | `outputs/` |
| 4 | `04_accuracy_area_estimation.py` | Olofsson accuracy and error-adjusted area | `outputs/accuracy_area_2023.txt` |
| 5 | `05_gaya_agreement_map.js` | Agreement with WorldCereal and MODIS | `gaya_agreement_*.csv` |
| 6 | `06_gaya_grace_groundwater.js` | GRACE minus GLDAS groundwater, and GLDAS-2.2 | `gaya_grace_monthly.csv` |
| 7 | `07_novelty_analysis.py` | Paired gap test, groundwater trend, driver Random Forest (spatial-block cross-validation) | `outputs_novelty/` |
| 8 | `08_yield_validation.py` | VHI against ICRISAT rice statistics | `outputs_yield/` |
| 9 | `09_cgwb_wells.py` | Parses the CGWB Ground Water Year Books (Bihar) and compares them with GRACE | `outputs_novelty/G_cgwb_*` |
| 10 | `10_regional_analysis.py` | Runs steps 3 and 7 for each district | `outputs_region/regional_summary.csv` |
| 11 | `11_inter_rater.py` | Agreement of repeat labelling (Cohen's κ); `python 11_inter_rater.py repeat` | `outputs_novelty/H_intra_rater.txt` |
| 12 | `12_revision_tests.py` | Drought and heat tests on the revision groups | `outputs_novelty/J_revision_tests.*` |
| 13 | `13_figure2_workflow.py` | Figure 2 (workflow) | `outputs/Figure2_workflow.png` |
| 14 | `14_figure1_exports.js` → `14b_figure1_study_area.py` | Figure 1 (study area) | `outputs/Figure1_study_area.png` |
| 15 | `15_figures_3_5.py` | Figures 3–5 | `outputs/Figure3–5_*.png` |
| 16–17 | `16_build_word_manuscript.py`, `17_cover_letter_docx.py` | Builds the submission files (need the manuscript, which is not in this repository) | — |
| 18 | `18_matching_equivalence.py` | Propensity-matched pixel comparison; equivalence test and minimum detectable effect | `outputs_novelty/K_matching_equivalence.*` |
| 19 | `19_sentinel2_kharif_droughts.js` | Sentinel-2 20 m check of the 2022–23 Kharif droughts | `outputs_novelty/L_sentinel2_kharif_droughts.txt` |
| 20 | `20_des_yield_validation.py` | DES district yields 1997–2022: rice in drought years; wheat, gram, lentil and mustard vs Feb–Mar heat | `outputs_yield/DES_yield_validation.*` |
| 21 | `21_district_comparability.py` | Drought severity, rainfall and VHI losses in the seven districts | `outputs_region/district_comparability.*` |

Earth Engine scripts export CSV files to the Google Drive folder `Gaya_Irrigation`; copy them into `data/` before running the Python scripts.

## Data not included
These third-party files are **not redistributed** here. Download them from the original sources and place them as shown:

| File(s) | Source | Place in |
|---|---|---|
| Ground Water Year Books, Bihar (2018–19 to 2020–21) | Central Ground Water Board, https://cgwb.gov.in | `data/cgwb/` |
| ICRISAT District Level Database (unapportioned, 1990–2015), and irrigated wheat area | ICRISAT DLD / Mendeley Data, doi:10.17632/ywp3y5j9vv.1 | `data/mendeley_icrisat/`, `data/icrisat_gaya_*.csv` |
| FAO GAUL 2015 boundaries | FAO, via Google Earth Engine (`FAO/GAUL/2015`) | exported by `14_figure1_exports.js` |

All satellite data (Sentinel-1/2, MODIS, CHIRPS, ERA5-Land, GRACE/GRACE-FO, GLDAS, ESA WorldCover and WorldCereal) are read directly in Google Earth Engine.

## Licence and citation
Code is released under the MIT licence (see `LICENSE`). If you use this code, please cite the paper (citation to be added on publication).
