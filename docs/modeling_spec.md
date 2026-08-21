# Modeling Specification

## Design: two linked stages, not one black-box model
Framing it as two stages keeps each model small, interpretable, and matched to the data volume we actually have — and it directly answers RQ1 and RQ2 separately instead of conflating them.

```
Stage A: Climate  ->  Discharge          Stage B: Discharge (+Climate)  ->  Coastal Chlorophyll
  SPI-1/3/6/12                              Discharge anomaly (lag 0-3mo)
  Monthly precip           -->  Q̂            SPI-3(t)                       -->  Chl-a anomaly
  Month-of-year (sin/cos)                    Month-of-year (sin/cos)
  Q anomaly(t-1) [persistence feature]
```

## Datasets & channels

| Stage | Input channels | Target channel | Source | Native resolution -> used at |
|---|---|---|---|---|
| A | SPI-1, SPI-3, SPI-6, SPI-12; monthly precip total; month-of-year (sin, cos); Q anomaly(t-1) | Discharge anomaly, log-space, monthly | Precip: NOAA GHCN-Daily station(s) near Caguas/San Juan. Discharge: USGS gauge 50055000 | Daily -> monthly |
| B | Discharge anomaly at lag 0,1,2,3 months; SPI-3(t); month-of-year (sin, cos) | Coastal chlorophyll-a anomaly, log-space, monthly | Discharge: Stage A gauge. Chlorophyll: satellite ocean-color monthly composite (OC-CCI merged product preferred over single-sensor MODIS/VIIRS — longer, bias-corrected record) for a coastal polygon near the Loíza river mouth | Daily/8-day -> monthly |

**Why anomalies, not raw values:** raw discharge and chlorophyll are dominated by the seasonal cycle. Subtracting each variable's own long-term monthly climatology (deseasonalizing) isolates the climate-driven signal we actually want to attribute — this is standard practice, not optional.

**Why log-transform:** streamflow and chlorophyll are both approximately log-normal (heavy right skew from flood/bloom events). Fit models in log-space, back-transform for reporting in native units.

## Training period

| Stage | Record length | Rationale |
|---|---|---|
| A | 1959–2025 (66 yr, ~792 monthly points) | Full unregulated Caguas discharge record; precip station record should cover a comparable span (verify overlap when pulled) |
| B | ~1998–2025 (bounded by ocean-color satellite record; OC-CCI blends SeaWiFS+MODIS+VIIRS from 1997) ≈ 27 yr, ~324 monthly points | Satellite chlorophyll is the limiting record — cannot be extended earlier |

Stage B is comparatively small — this drives the model-complexity decision below.

## Model choice — and why not deep learning
With a few hundred monthly points, an LSTM or any deep model will overfit; the sample size doesn't support it. Match model complexity to data volume:
1. **Baselines (mandatory, run first):** climatological mean, persistence (t-1)
2. **Interpretable baseline:** multiple linear regression / ElasticNet (standardized features)
3. **Main model:** Random Forest and/or Gradient Boosted Trees (XGBoost) — handles nonlinearity/interaction without needing thousands of samples, and SHAP attribution works natively on trees
4. Deep learning is explicitly out of scope for this stage — noted as a possible future direction only if we later pool multiple watersheds into one larger multi-basin dataset

## Training method & folding
- **Stage A** (large-n): chronological split — train 1959–2013 (~80%), holdout test 2014–2025 (~20%), never touched during tuning. For hyperparameter tuning within the training period, use `TimeSeriesSplit` (expanding window, 5 folds) — no random shuffling, no standard k-fold (would leak autocorrelated neighbors across train/test).
- **Stage B** (small-n): single holdout is too noisy with ~324 points. Use expanding-window time-series CV (4–5 folds) across the whole record and report **mean ± spread of NSE/RMSE across folds**, not a single number.
- Hyperparameters tuned only within training folds; test/holdout touched exactly once, at the end, for final reported numbers.

## Evaluation (full detail in `validation_protocol.md`)
NSE, RMSE, MAE on held-out data only, benchmarked against climatology/persistence; SHAP for driver/lag attribution; sanity-check attributed lag against physically plausible basin transit time.

## Preprocessing pipeline (implementation order)
1. Discharge: daily mean (USGS `00060`) → monthly mean → log → subtract monthly climatology = anomaly
2. Precipitation: daily total (GHCN) → monthly sum → SPI-1/3/6/12 via gamma-distribution fit (`climate_indices` package, fit on full record for stable parameters)
3. Chlorophyll: satellite monthly composite over Loíza-mouth coastal polygon → spatial mean → log → subtract monthly climatology = anomaly
4. Join all series on `YYYY-MM`; gaps <2 months linearly interpolated, longer gaps dropped (log how many rows dropped and why — e.g., persistent cloud cover)
5. Scale features for the linear baseline only; tree models use raw (anomaly, log) values directly
