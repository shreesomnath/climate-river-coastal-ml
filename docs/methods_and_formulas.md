# Methods & Formulas Reference

Every formula/method used in this project, why it was chosen, and where it's implemented. Written so it can be lifted almost directly into a Methods section later.

## 1. Standardized Precipitation Index (SPI)

**What it is**: a drought index that expresses how anomalous a given precipitation total is, in standard-deviation units, relative to the long-term climatology for that same accumulation window and calendar month.

**Formula** (McKee, Doesken & Kleist, 1993 — the original, still-standard SPI method):
1. Accumulate precipitation over a rolling window of *w* months (we use w = 1, 3, 6, 12).
2. For each calendar month separately, fit a two-parameter Gamma distribution to the accumulated totals (shape α, scale β), via maximum likelihood.
3. Convert each observed accumulated total to a cumulative probability via the fitted Gamma CDF (with a separate correction for the probability of zero precipitation, since Gamma is undefined at 0).
4. Transform that probability to a standard normal Z-score via the inverse normal CDF (probit). That Z-score *is* the SPI value.

**Why Gamma + per-calendar-month fitting**: precipitation is not normally distributed (right-skewed, bounded at zero); fitting Gamma separately per calendar month accounts for PR's strong wet/dry seasonality without needing a separate deseasonalizing step first.

**Where**: `scripts/03_preprocess_climate_discharge.py`, function `spi()`.

**Significance in this study**: SPI-1 (short-term) turned out to be the dominant driver of discharge anomaly (SHAP, Stage A) — the basin responds to drought/wet conditions within about a month, not on a multi-year lag. This ruled out the initially-considered ENSO-driven framing (literature review found ENSO is *not* a strong direct PR rainfall driver) in favor of local SPI as the primary climate variable.

## 2. Log-transform + deseasonalized anomaly

**What/why**: streamflow and chlorophyll-a are both approximately log-normal (heavy right skew from flood/bloom events) — models are fit in log-space. Each series' own long-term mean for that calendar month is then subtracted, isolating the year-to-year climate-driven signal from the (much larger) seasonal cycle:

```
anomaly(t) = log(x(t)) - mean(log(x(t')) for all t' in the same calendar month across the record)
```

**Where**: `03_preprocess_climate_discharge.py` (discharge), `07_preprocess_stage_b.py` / `13_add_sst_wind_covariates.py` (chlorophyll, SST, wind).

## 3. Inter-sensor bias correction (MODIS <-> VIIRS chlorophyll merge)

**Problem**: no single satellite ocean-color sensor covers the full discharge record (1959-present). MODIS-Aqua (legacy CoastWatch product) covers 2003-2022; VIIRS covers 2012-present. Naively concatenating the two would introduce a step-change artifact at the switch point, since each sensor/algorithm has its own retrieval bias.

**Method**: standard practice for merging ocean-color products (the same principle OC-CCI uses at a global scale). On the overlap period (121-121 months depending on box), fit:
```
log(MODIS) = a + b * log(VIIRS)      (ordinary least squares)
```
Apply this correction to VIIRS values *only for months after MODIS ends* (no double-adjustment in the overlap window; MODIS is used natively there).

**Where**: `09_merge_chlorophyll_sensors.py` (original box), `12_stage_b_far_offshore.py` (far_offshore box, inline).

**Result**: R²=0.484 (original box) to 0.688 (far_offshore box) on the overlap regression — the far_offshore box's tighter fit is itself evidence its chlorophyll signal is spatially more consistent/less noisy.

## 4. Evaluation metrics

| Metric | Formula | Why used |
|---|---|---|
| **NSE** (Nash-Sutcliffe Efficiency) | `1 - sum((obs-pred)^2) / sum((obs-mean(obs))^2)` | The standard hydrology skill score. NSE=1 perfect, NSE=0 equals just predicting the mean (climatology), NSE<0 worse than that. Primary metric throughout. |
| **KGE** (Kling-Gupta Efficiency) | `1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)` where r=correlation, alpha=std(pred)/std(obs), beta=mean(pred)/mean(obs) | Decomposes skill into correlation/variability/bias components — more diagnostic than NSE alone. **Caveat found during this study**: beta is numerically unstable for zero-centered anomaly series (small denominator), so KGE is reported but NSE/Spearman are treated as primary for the anomaly-based Stage B models. |
| **RMSE, MAE** | standard | Native-unit interpretability alongside the skill scores. |
| **Spearman ρ** | rank correlation | Robust to the exact NSE/KGE calibration issues above; tells us whether the model ranks high/low periods correctly even when it doesn't nail magnitudes. Found to be more favorable than NSE for Stage B, i.e. the model is better at relative timing than absolute magnitude. |
| **Block bootstrap CI** | resample contiguous 6-month blocks (not individual points) 1000x, take 2.5/97.5 percentiles of NSE | Individual-point bootstrap would violate the autocorrelation in monthly climate/ocean data and understate uncertainty; block size (6 months) chosen to be longer than the SHAP-observed lag-decay (~3 months). |

## 5. Cross-correlation / lag analysis

Simple Pearson correlation between one series and a lagged version of another, scanned across a range of lags, to find both the strongest linkage and its characteristic timescale. Used to (a) sanity-check that Stage A's SPI-discharge link peaks at a physically plausible lag (0 months — fast basin response), and (b) discover the far_offshore box's peak discharge-chlorophyll link, also at lag 0.

## 6. Models

- **Baselines (always fit first)**: climatology (predict the calendar-month mean, i.e. 0 for an anomaly series) and persistence (predict last month's observed value). A model that can't beat these has no claim to skill.
- **ElasticNet** (linear, L1+L2 regularized, `sklearn.ElasticNetCV`): the primary model. Chosen because — confirmed repeatedly across Stage A and Stage B — it matches or beats RandomForest at this sample size (n in the hundreds), consistent with domain literature where deep/complex models only show an edge with much larger (daily/high-frequency) records.
- **RandomForest** (`n_estimators=300-400, max_depth=4-6`): fit alongside ElasticNet as a nonlinearity check and for its native SHAP compatibility. Shallow max_depth deliberately constrains complexity to match the data volume.
- **Deep learning (LSTM etc.)**: deliberately not used. Explicit decision, not an oversight — see `results_summary.md` "On better models."

## 7. Explainability: SHAP

`shap.TreeExplainer` on the fitted RandomForest, mean |SHAP value| per feature on held-out data, to answer *which* driver and at what lag — not just whether the model predicts well. (Lundberg & Lee, 2017, "A Unified Approach to Interpreting Model Predictions.")

## 8. Validation folding — two independent schemes, deliberately

1. **Expanding-window `TimeSeriesSplit`**: train on an ever-growing prefix, test on the next block. Standard for time series — never trains on the future.
2. **Contiguous blocked K-Fold** (`shuffle=False`): splits the record into 5 contiguous chronological blocks, rotating which block is held out. Added specifically so no single fold-scheme's idiosyncrasies could be mistaken for a real result — used to independently confirm the far_offshore box's positive skill.

Never used: random/shuffled K-fold — would leak autocorrelated neighboring months across train/test and overstate skill.
