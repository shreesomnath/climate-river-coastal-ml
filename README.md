# Climate-Driven River–Coastal Connectivity Across Puerto Rico: Multi-Basin Evidence from Hydrologic and Satellite Observations

A multi-basin study of the climate → river discharge → coastal chlorophyll-a chain in Puerto Rico, built entirely on public USGS/NOAA/satellite data, using explainable machine learning as the analytical tool. This repository contains the full, reproducible data pipeline behind the accompanying manuscript (prepared for *Journal of Hydrology: Regional Studies*).

**Authors:** Somnath Luitel, Manmeet Singh — AI Research Lab, Department of Earth, Environmental, and Atmospheric Sciences, Western Kentucky University

## Overview

Puerto Rico's short, steep watersheds respond quickly to rainfall variability, and the resulting river discharge reaches the coast within a short distance and time. This project tests three hypotheses across 10 independent Puerto Rico watersheds:

- **H1** — Local drought conditions (Standardized Precipitation Index), not the El Niño–Southern Oscillation, are the dominant driver of river discharge variability.
- **H2** — A drought-driven discharge signal detectably propagates to coastal chlorophyll-a in at least some watersheds.
- **H3** — An explainable ML framework (SHAP) can attribute predictive skill to a specific driver and lag, and diagnose *why* a linkage is or isn't detected in a given basin.

**Headline results** (see `manuscript/manuscript.pdf` for full detail):
- **Stage A (climate → discharge)**: confirmed in all 10 tested basins (ElasticNet NSE 0.32–0.74; primary basin NSE = 0.390, permutation-test *p* = 0.005). SHAP attributes this to 1-month SPI, consistent with fast basin hydrological response.
- **Stage B (discharge → coastal chlorophyll-a)**: significant in 2 of 7 testable basins (*p* = 0.005 for both), weak/mixed in 2 more, null in 3. Cross-basin variation is partly explained by MODIS↔VIIRS satellite sensor-merging quality — a confound identified and directly confirmed by MODIS-only reprocessing in 2 of 3 weak-correction basins.

## Repository structure

```
.
├── scripts/                    # Numbered, sequential pipeline (01 → 32)
│   ├── 01-14                   # Single-basin (Loíza) pilot: fetch, preprocess, model, map
│   ├── 15-29                   # Multi-basin expansion: batch fetch/model, diagnostics
│   ├── 30-31                   # Rigor checks: permutation tests, learning curves
│   ├── 32                      # Methodology flowchart figure
│   └── plot_style.py           # Shared matplotlib style
├── data/
│   ├── raw/                    # Fetched source data (git-ignored — regenerate via scripts)
│   └── processed/              # Model-ready tables and saved results (small, versioned)
├── figures/                    # All publication figures (PNG, 300 dpi)
├── docs/                       # Working notes: methods reference, literature review,
│                                #   validation protocol, results summary
├── manuscript/
│   ├── manuscript.tex          # Elsevier elsarticle-format source
│   ├── manuscript.pdf          # Compiled PDF
│   ├── manuscript.docx         # Word version (for journal submission / collaborator edits)
│   └── references.bib          # Verified bibliography
├── requirements.txt
└── LICENSE
```

## Data sources

All data are public and require no authentication.

| Variable | Source | Access |
|---|---|---|
| River discharge | USGS National Water Information System (NWIS) | REST API |
| Precipitation | NOAA Global Historical Climatology Network-Daily (GHCN-D) | REST API |
| Chlorophyll-a | NOAA CoastWatch ERDDAP — MODIS-Aqua (2003–2022) + VIIRS/S-NPP (2012–present) | ERDDAP griddap |
| Sea-surface temperature | NOAA OISST v2.1 | ERDDAP griddap |
| Wind speed | NCEP/NCAR Reanalysis | ERDDAP griddap |
| River flowlines / mouths | USGS National Hydrography Dataset; OpenStreetMap (fallback) | REST API / Overpass |
| Coastline boundary | UN OCHA Common Operational Dataset (COD-AB), 2019 | Static shapefile |
| Elevation | GEBCO_2020 (15 arc-second) | ERDDAP griddap |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a working internet connection to fetch data (all sources above are open/unauthenticated). A LaTeX distribution (e.g., TeX Live, with `elsarticle.cls`) and `pandoc` are needed only to rebuild the manuscript PDF/DOCX.

## Reproducing the pipeline

Scripts are numbered in execution order. The single-basin pilot (Río Grande de Loíza) runs first; the multi-basin expansion and rigor checks build on it.

```bash
# Single-basin pilot
python scripts/01_fetch_discharge.py
python scripts/02_fetch_precip.py
python scripts/03_preprocess_climate_discharge.py
python scripts/04_model_stage_a.py
python scripts/05_fetch_chlorophyll.py
python scripts/06_fetch_chlorophyll_viirs.py
python scripts/07_preprocess_stage_b.py
python scripts/08_model_stage_b.py
python scripts/09_merge_chlorophyll_sensors.py
python scripts/10_study_area_map.py
python scripts/11_test_coastal_boxes.py
python scripts/12_stage_b_far_offshore.py
python scripts/13_add_sst_wind_covariates.py
python scripts/14_metrics_comparison_plots.py

# Multi-basin expansion (9 additional watersheds)
python scripts/15_batch_fetch_watersheds.py
python scripts/16_batch_find_precip_stations.py
python scripts/17_batch_preprocess_and_model_stage_a.py
python scripts/18_common_window_sensitivity.py
python scripts/19_batch_test_coastal_boxes.py
python scripts/20_stage_b_top4_full_validation.py
python scripts/21_multibasin_final_synthesis.py
python scripts/22_full_metrics_all_basins.py
python scripts/23_basin_characteristics_analysis.py
python scripts/24_basin_characteristics_plot.py
python scripts/25_add_3_more_watersheds.py
python scripts/26_sst_wind_more_basins.py
python scripts/27_bias_r2_vs_stage_b_plot.py
python scripts/28_modis_only_retest.py
python scripts/29_modis_only_comparison_plot.py

# Statistical rigor checks
python scripts/30_significance_and_leakage_checks.py
python scripts/31_permutation_and_learning_curves.py

# Figures
python scripts/32_methodology_flowchart.py
```

`data/raw/` is git-ignored (regenerated by the fetch scripts above); `data/processed/` contains the small, versioned intermediate tables and saved model results that the manuscript's figures and tables are built from directly.

## Methodology

Two-stage explainable ML framework:
1. **Stage A**: SPI (1/3/6/12-month, gamma-distribution fit per calendar month) + seasonality + lag-1 discharge → discharge anomaly.
2. **Stage B**: discharge anomaly (lags 0–3) + SPI-3 + seasonality [+ SST, wind] → coastal chlorophyll-a anomaly.

Both stages: ElasticNet and Random Forest vs. climatology/persistence baselines, validated with two independent cross-validation schemes (expanding-window and blocked K-fold), block-bootstrap 95% CIs, circular-shift permutation significance tests, and learning-curve stability checks. SHAP provides driver/lag attribution. Full derivations (SPI gamma-CDF formulation, NSE, KGE) are in `manuscript/manuscript.tex` §3 and `docs/methods_and_formulas.md`.

## Manuscript

To rebuild the PDF and DOCX after editing `manuscript/manuscript.tex`:

```bash
cd manuscript
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
pandoc manuscript.tex --bibliography=references.bib \
  --csl=https://raw.githubusercontent.com/citation-style-language/styles/master/elsevier-harvard.csl \
  --citeproc -o manuscript.docx
```

## License

Code is released under the [MIT License](LICENSE). Data are redistributed or re-fetched from the public sources listed above under their respective terms of use.
