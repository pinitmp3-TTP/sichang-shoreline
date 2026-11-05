# Sichang shoreline: reference-calibrated optical–SAR shoreline change

Code and derived results for:

> P. Tanachaichoksirikun *et al.*, "Detecting human-made shoreline change on a data-scarce rocky island: A reference-calibrated optical–SAR approach," submitted to *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*.

The pipeline extracts waterlines from Landsat 5/7/8/9, Sentinel-2 and Sentinel-1 in Google Earth Engine (1988–2026), corrects them for tide with a harmonic model, casts shore-normal transects, harmonises sensors, computes linear regression rates, sets an empirical precision floor from an undisturbed reference cliff, and runs a blind single-sensor step scan to detect and date abrupt change.

## Contents

| File | Purpose |
|---|---|
| `sites.json` | Single configuration: site boxes, types, beach slopes, Sentinel-1 geometry, Drive folder. Set `ee_project` to your own Earth Engine project. |
| `00_data_inventory.py` | Counts scenes per sensor and picks the Sentinel-1 geometry |
| `01_parse_tides.py`, `02_fit_tides.py` | Digitise the tide table and fit the UTide harmonic model (trend disabled) |
| `03_predict_at_overpass.py` | Tide level at each acquisition time |
| `04_waterline_gee.py` | Waterline extraction in Earth Engine (MNDWI/Otsu; VV/Otsu), exported to Google Drive |
| `05_transect_rates.py` | Transects, tide correction, sensor harmonisation, LRR, breakpoint and blind step scan |
| `06_driver_attribution.py` | Backshore land-cover covariate (ESA WorldCover 2021) |
| `07_prepare_site.py` | Merge exports, quality control, baseline construction |
| `08_backshore_change_gee.py`, `09_timing_check.py` | Tested and rejected temporal land-cover layer (kept for transparency) |
| `10_estimate_slope.py` | Beach slope from the tide–position regression |
| `11_vessels_s1_gee.py`, `12_vessel_analysis.py` | Vessel presence from Sentinel-1 and comparison with shoreline change |
| `13_disturbance_area.py` | Disturbance footprints from digitised polygons |
| `run_pipeline.py` | Runs 07 → 03 → 05 → 06 for all sites and writes the summary |
| `Sichang_Colab.ipynb` | Step-by-step notebook for Google Colab |
| `sichang_tide_coef.pkl` | Fitted UTide coefficients for Ko Si Chang |
| `results_v1/` | Results used in the paper: `site_summary.csv`, and per site the transect rates (`*_rates.csv`), transects with rates (`*_transects.geojson`), baseline and scenes per year |

## How to run

1. Open `Sichang_Colab.ipynb` in Google Colab and put this folder in Google Drive.
2. Set `ee_project` in `sites.json`.
3. Run the cells in order: Earth Engine exports (04) go to `My Drive/<drive_folder>`; when they finish, `run_pipeline.py` computes rates and the summary.

Requirements are listed in `requirements.txt`.

## Data not included

* The Ko Si Chang tide table is published by the Hydrographic Department, Royal Thai Navy, and is not redistributed here; only the fitted harmonic coefficients are included.
* Reference waterlines were traced on Google Earth imagery and are not redistributed, in line with Google's terms; they are available from the authors on request.
* Satellite imagery is freely available from USGS (Landsat) and the Copernicus programme (Sentinel-1/2) through Google Earth Engine.

## Licence and citation

Code: MIT licence (see `LICENSE`). Results: CC BY 4.0. Please cite the paper and the archived version of this repository (DOI on the Zenodo badge / `CITATION.cff`).
