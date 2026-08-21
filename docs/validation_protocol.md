# Validation, Ground Truth, and Evaluation Protocol

This is the part most ML-in-hydrology abstracts skip or hand-wave. Written down now so it's not improvised later.

## 1. Where "ground truth" actually exists vs. doesn't

| Variable | Is it ground truth? | Notes |
|---|---|---|
| USGS river discharge (Caguas gauge) | **Yes** — direct sensor measurement, not a model output | This is real, no validation needed on our end; USGS already QA/QC's it (codes: A=approved, P=provisional) |
| Precipitation / SPI | **Yes**, if from NOAA GHCN station gauges | SPI is a derived index but computed from real rain-gauge totals, standard formula, no fitting involved |
| Satellite chlorophyll-a (MODIS/VIIRS) | **No — it's a retrieval, not truth** | This is the honest caveat: ocean-color chlorophyll is itself an algorithm output (e.g., OCx band-ratio) calibrated against global in-situ chlorophyll (SeaBASS). We have no local in-situ PR chlorophyll to re-validate it ourselves. |

**How to handle the chlorophyll caveat honestly (don't oversell):**
- Do not claim to "validate" satellite chlorophyll accuracy — that's NASA/NOAA's job, already done at the algorithm level (cite their cal/val documentation).
- Use satellite chlorophyll in **relative/anomaly form** (deviation from its own multi-year monthly climatology at that pixel/region), not absolute concentration — anomalies are more robust to retrieval bias than absolute values.
- State this explicitly in the abstract/limitations: "satellite-derived chlorophyll-a used as a proxy signal, not validated in situ."

## 2. What actually needs our own validation: the ML model

This is the part we control and must do rigorously.

### a. Train/test split — time-series aware, not random
Random k-fold shuffling on time series data leaks future information into training via autocorrelation (a well-known failure mode in hydro-ML papers). Instead:
- **Blocked temporal split**: train on years 1–N (e.g., ~80% of the record), hold out the most recent ~20% of years as test.
- If doing cross-validation for hyperparameter tuning, use `sklearn.model_selection.TimeSeriesSplit` (expanding window), never standard `KFold`.

### b. Baseline comparison — mandatory
An ML model that isn't compared to a naive baseline proves nothing. Baselines to report alongside RF/XGBoost:
- **Climatological mean**: predict each month's chlorophyll anomaly as 0 (i.e., the long-term monthly climatology) — this is the standard hydrology null model.
- **Persistence**: predict this month = last month's observed value.
- The ML model must beat both to claim skill.

### c. Metrics (hydrology-standard, not just R²)
- **NSE (Nash–Sutcliffe Efficiency)** — the standard hydrology skill score; NSE=1 is perfect, NSE=0 equals the mean baseline, NSE<0 is worse than just guessing the mean.
- RMSE, MAE for interpretability in native units.
- Report metrics on the **held-out test period only** — training-set performance is not evidence.

### d. Feature importance / explainability, not just accuracy
- Use SHAP values on the fitted RF/XGBoost model to show *which* lagged climate/discharge feature actually drives predictions — this is what lets us answer "which driver and what lag," not just "can we predict."
- Sanity-check SHAP results against physical plausibility (e.g., does the model pick up a lag that's physically reasonable for water transit time through the basin, ~days to weeks, not e.g. a 5-year lag which would be a red flag for a spurious correlation).

### e. Uncertainty
- Bootstrap or repeated time-blocked CV to get a spread on NSE/RMSE rather than a single point estimate — report as a range, not a bare number.

## 3. What this means for the abstract (Sept 15)
We don't need final polished numbers by the deadline — we need to show the pipeline runs end-to-end on real data with a defensible train/test protocol and at least a preliminary NSE vs. baseline comparison. That preliminary number, even if modest, is what makes the abstract's "results" sentence honest rather than hand-wavy.
