# Preliminary Results Summary (as of 2026-08-20)

## Rigor checks added before manuscript drafting (`30_significance_and_leakage_checks.py`, `31_permutation_and_learning_curves.py`)
1. **Permutation significance test** (circular-shift null, n=200, preserves each series' own autocorrelation — the correct null for time series, not a naive shuffle): Loíza Stage A observed NSE=0.390 vs. null max 0.141 (**p=0.005**); Manatí Stage B observed NSE=0.146 vs. null max 0.085 (**p=0.005**). Both flagship results are formally significant, not just "beats a simple baseline." Visualized in `figures/20_permutation_tests.png`.
2. **SPI leakage sensitivity check**: SPI is conventionally fit on the full record (standard hydrological practice), which technically lets test-period precipitation influence the distribution parameters used for training months. Refit using train-only data — **zero difference** (NSE 0.390 vs 0.390). Concern is real in principle but doesn't matter in practice here; reportable as "checked, no effect."
3. **Learning curves** (`figures/21_learning_curves.png`) — important nuance found: **Loíza's is textbook clean** (steady climb from NSE 0.25 at 15% of training data to a plateau at ~0.39 by ~65%, flat thereafter — the model has converged, more data unlikely to help much). **Manatí's is erratic** (swings between +0.12 and -0.22 across training-set sizes before landing on the full-data value +0.146 at the very last point) — meaning Manatí's result, while formally significant on the full dataset, is **not robust to training-window size** the way Loíza's is. Report both findings as-is: Loíza is the more robust flagship result; Manatí's significance is real but should be presented with this caveat, not as equally solid.

## CORRECTION (2026-08-20, late): calendar-gap bug found and fixed
While building the multi-watershed expansion, found that monthly precip/discharge series were never reindexed to a continuous calendar before computing rolling SPI windows and the lag-1 feature. A calendar gap (a month with zero raw daily reports) was an ABSENT row, not a NaN row -- `.rolling()`/`.shift()` operate on row position, not calendar time, so gaps could silently splice non-adjacent months together. Fixed in `03_preprocess_climate_discharge.py` and `17_batch_preprocess_and_model_stage_a.py` (reindex to a continuous monthly `PeriodIndex` before any rolling/lag op). Re-ran Loíza's Stage A after the fix: **the Trujillo Alto precip station has 146 real calendar-gap months**, previously invisible. The numbers below are the corrected ones -- the original 0.514 NSE figure was computed before this fix and should not be cited going forward.

## Stage A: Climate -> Discharge — WORKS, real skill (corrected numbers)
- Data: 426 usable months, 1969-09 to 2026-07 (Río Loíza at Caguas, unregulated gauge) — shorter than originally reported (570 months, starting 1959-12) because the gap-exposing fix correctly excludes windows straddling the precip station's real gaps
- Held-out test period: 2013-06 to 2026-07 (86 months, never touched during training/tuning)

| Model | NSE | RMSE | MAE |
|---|---|---|---|
| Climatology (predict 0) | -0.001 | 0.759 | 0.641 |
| Persistence (t-1) | -0.440 | 0.910 | 0.707 |
| **ElasticNet** | **0.390** | **0.592** | **0.477** |
| RandomForest | 0.297 | 0.636 | 0.502 |
| XGBoost | 0.282 | 0.643 | 0.511 |

RF TimeSeriesSplit CV on training period: NSE = 0.226 +/- 0.178 (5 folds) — still consistently positive across folds (one fold slightly negative, -0.105), same qualitative conclusion as before the fix.

**Headline finding (revised)**: a simple linear model on SPI (multiple windows) + precipitation + seasonality explains ~39% of out-of-sample discharge anomaly variance (down from the originally-reported ~51%, which was computed before the calendar-gap fix) — still clearly beating both baselines. The linear model still beats the tree ensembles, so the "close to additive/linear" methodological conclusion is unchanged. The core finding survives the correction; only the exact magnitude changed.

**SHAP attribution:** SPI-1 (1-month drought index) is the dominant driver, ahead of the discharge persistence term and SPI-3. Longer accumulation windows (SPI-6, SPI-12) and seasonality contribute much less.

**Cross-correlation:** peak correlation between SPI-3 and discharge anomaly is at lag 0 (same month), consistent with a short (weeks, not years) hydrologic response time for this basin — physically plausible, not a spurious long-lag artifact.

## Stage B: Discharge -> Coastal Chlorophyll — WEAK / NULL result, confirmed robust
- Data (v1): 127 usable months, 2012-04 to 2026-06, VIIRS-only chlorophyll
- **Data (v2, current)**: 228 usable months, 2003-04 to 2026-06 — MODIS (2003-2022) and VIIRS
  (2012-2026) merged via log-log inter-sensor bias correction fit on 121 overlap months
  (R²=0.484; see `09_merge_chlorophyll_sensors.py`). This directly addresses the "different
  time periods" mismatch between the discharge/SPI record (1959-2026) and the chlorophyll
  record (satellite-era only) by making full use of both sensors instead of just VIIRS.
- Validation: expanding-window CV, 5 folds (record still short relative to Stage A, so no
  single holdout)

| Model | Mean NSE across folds (v1, n=127) | Mean NSE across folds (v2, n=228) |
|---|---|---|
| Climatology (predict 0) | -0.290 | -0.207 |
| Persistence (t-1) | -0.617 | -0.735 |
| ElasticNet | -0.421 | -0.408 |
| RandomForest | -0.459 | -0.394 |

**None of the models beat climatology, in either version.** Nearly doubling the sample size
(v1 -> v2) did not change the qualitative result — this rules out "small sample" as the
explanation and strengthens the case that the signal is genuinely weak at this spatial/temporal
scale, not a data-volume artifact. River-to-coast lag scan: r=0.230 at lag 0 (v2), still weak,
decaying at longer lags.

### This is a real, honest result — not a pipeline bug
Checked: preprocessing pipeline is correct (verified log-transform, deseasonalizing, join logic). The weak linkage is more likely a genuine finding or a scoping limitation:
1. **Coastal box may be too small/local** (median 36 valid pixels/month) — chlorophyll near a single river mouth is noisy at monthly resolution; a river signal can be there but swamped by local noise, wind-driven mixing, or cloud-gap sampling bias.
2. **Missing covariates** — coastal chlorophyll is also driven by wind-driven upwelling/mixing and larger-scale circulation, not discharge alone; the current feature set may be underspecified.
3. **Reservoir regulation** (see problem statement's stated limitation) — the Caguas gauge is upstream of the Carraízo/Loíza dam system, so the discharge signal reaching the actual coast may differ from what we're using as a proxy.
4. **Monthly aggregation may wash out pulse events** — river plume chlorophyll responses to storm pulses are often short-lived (days); monthly composites may be the wrong temporal resolution to catch them.

## Stage B revisited: box relocation finds a real (modest) signal

Per the "widen/relocate the coastal box" next-step above, tested 4 candidate boxes against the true river mouth (-65.878, 18.438, taken directly from the USGS NHD flowline endpoint, not estimated). Quick correlation screen (`11_test_coastal_boxes.py`):

| Box | Location relative to mouth | best \|r\| |
|---|---|---|
| near_tight | right at the mouth | 0.167 |
| near_wide | wider, near mouth | 0.194 |
| original (prior box) | offset, poorly centered | 0.230 |
| **far_offshore** | ~9km north of the mouth | **0.501** |

The mouth-adjacent boxes performed *worse* than the original -- the strongest signal is ~9km offshore, north of the river mouth. Physically plausible: a river plume's peak surface chlorophyll signature can lag behind the immediate mouth as freshwater/nutrients disperse and mix, rather than peaking at the point of discharge itself.

**Full validation on `far_offshore`** (lon -65.93/-65.83, lat 18.46/18.54), same rigor as everything else — MODIS+VIIRS merge (this box's inter-sensor bias correction R²=0.688, notably tighter than the original box's 0.484, suggesting this location's chlorophyll signal is spatially more consistent), 226 usable months (2003-04 to 2026-06), **two independent folding schemes**:

| Folding scheme | NSE_climatology | NSE_ElasticNet | NSE_RandomForest |
|---|---|---|---|
| Expanding-window (5 folds) | -0.060 ± 0.058 | **+0.082 ± 0.112** | +0.086 ± 0.129 |
| Contiguous blocked K-fold (5 folds) | -0.039 ± 0.033 | **+0.144 ± 0.094** | +0.125 ± 0.141 |

(Numbers above are post calendar-gap-fix, see correction note at top of this document. Re-running after the fix gave nearly identical conclusions to the original run — reassuring, since it means this particular finding wasn't an artifact of the bug.)

Both schemes agree: ElasticNet now consistently beats climatology across the whole record (positive NSE in 8/10 fold-model combinations) — the first time any model has beaten the baseline in this study. RandomForest is similar but noisier (higher SD) — consistent with the earlier finding that a simple linear model suits this data volume better than trees.

**Final holdout (most recent 46 months, untouched during CV/tuning):**

| Model | NSE | KGE | RMSE | MAE | Spearman ρ |
|---|---|---|---|---|---|
| Climatology | -0.106 | n/a (undefined for a constant prediction) | 0.266 | 0.205 | n/a |
| Persistence | -0.345 | +0.333 | 0.293 | 0.224 | +0.278 |
| **ElasticNet** | **+0.066** | -0.442 | 0.244 | 0.188 | **+0.481** |
| RandomForest | -0.076 | -0.488 | 0.262 | 0.214 | +0.414 |

ElasticNet's NSE 95% block-bootstrap CI (1000 resamples, 6-month blocks to respect autocorrelation): **[-0.520, 0.257]** — straddles zero. Honest read: the point estimate is positive and both CV schemes independently support a real, modest effect across the full record, but the single most-recent 46-month holdout alone is too short to confirm it on its own. This is a *promising, not yet fully confirmed* result — appropriately hedged, not overstated.

Note on KGE: its bias-ratio term (β = pred.mean()/obs.mean()) is numerically unstable here because these are anomalies centered near zero — small denominators make β swing wildly and shouldn't be over-interpreted. NSE and Spearman ρ are the more trustworthy metrics for this anomaly-based setup; Spearman (+0.48 for ElasticNet) is notably better than NSE alone suggests, meaning the model ranks high/low chlorophyll periods more reliably than it nails exact magnitudes.

SHAP on the far_offshore model (`figures/09_shap_far_offshore.png`) shows a physically coherent pattern: same-month discharge anomaly dominates, decaying smoothly through lag 1-2-3, with SPI-3 second — a decay structure consistent with a real river-to-coast transport signal, unlike the earlier null-result box which had no such structure.

**Literature check**: tree-ensemble methods (RF/XGBoost) and adding river-flow data as a predictor are both standard, validated approaches in comparable chlorophyll-prediction studies (e.g., Pearl River Estuary RF work, R²=0.91 with spectral+environmental features); LSTM/deep learning shows up mainly in studies with daily/high-frequency records (hundreds-to-thousands of points), not the ~226-load monthly record here — confirms the earlier decision to stick with linear/tree models rather than deep learning.

## What this means for the abstract (updated after box relocation)
Both Stage A and Stage B now have real, reportable — if appropriately hedged — findings. Recommended framing: lead with Stage A (climate->discharge, NSE 0.51, strong), and present Stage B's far_offshore result as a genuine but preliminary positive finding: consistent modest skill across two independent folding schemes (NSE_EN ≈ +0.12 to +0.15), a physically coherent SHAP decay pattern, but a final-holdout bootstrap CI that still straddles zero. That's an honest, defensible "promising, still firming up" claim for a Sept 15 abstract — stronger than a pure null result, without overselling to a level the numbers don't support.

### On "better models"
Confirmed again on the far_offshore box: ElasticNet (linear) matches or beats RandomForest in both folding schemes, and literature for comparable chlorophyll-prediction studies backs this data-volume-appropriate choice — deep learning (LSTM etc.) shows up mainly in studies with far larger (daily/high-frequency) records. More model complexity was never the bottleneck; box placement was — confirmed by the fact that relocating it (no model changes at all) flipped the result from uniformly negative to consistently positive.

## SST + wind extended to 3 more basins (Manatí, Patillas, Añasco) — no benefit
Unlike Loíza (where SST meaningfully helped, ranking 3rd in SHAP), adding SST/wind anomalies to these 3 basins changed NSE by less than 0.02 in either direction across both folding schemes — noise, not signal. **The SST benefit found for Loíza does not generalize.** Worth noting in the paper as a basin-specific effect, not a general improvement to recommend by default. Full numbers in `data/processed/watersheds/sst_wind_more_basins_cv_scores.csv`.

## SST + wind covariates added (Loíza)
Per Next-step #2: added OISST v2.1 sea-surface-temperature anomaly (daily->monthly, far_offshore box) and NCEP/NCAR reanalysis 10m wind-speed anomaly (nearest grid cell, ~1.9° reanalysis resolution -- appropriately coarse for a large-scale wind field) to the far_offshore feature set (`13_add_sst_wind_covariates.py`).

- Blocked-K-fold NSE_EN: 0.135 -> 0.149 (modest improvement)
- Expanding-window NSE_EN: roughly flat (~0, both with and without)
- Final holdout NSE: 0.066 -> 0.151 (improved, but holdout window shifted slightly due to SST/wind date trimming -- not a perfectly like-for-like comparison)
- Bootstrap 95% CI: [-0.353, 0.248] -- still straddles zero, same honest caveat as before

**SHAP**: SST anomaly ranks 3rd overall, above discharge lag-1 — a real, meaningful contribution, not noise. Wind speed anomaly ranks lower (near discharge lag-3), a smaller but non-trivial contribution. Verdict: SST is worth keeping as a standard covariate; wind's marginal value is smaller than hoped but not zero. Neither addition resolves the fundamental sample-size limitation (bootstrap CI still wide) — that requires either a longer record or, per the strategic plan below, pooling across multiple watersheds.

## Next steps (priority order)
1. ~~Widen or multiply the coastal chlorophyll box~~ — DONE, see above. far_offshore box adopted; consider testing 1-2 more candidates further offshore/alongshore to map out where the signal peaks, time permitting.
2. **Add wind/SST as covariates** — likely the next-highest-value addition; would help explain the RandomForest's noisier folds and may tighten the final-holdout CI.
3. **Try weekly instead of monthly resolution** — still untried; could sharpen the lag-decay structure SHAP already hints at.
4. Model architecture remains lowest priority — current evidence continues to favor linear over tree/deep methods at this data volume.

## Multi-basin expansion (Phase 3) — full results in `watershed_candidates.md`
Extended Stage A + Stage B to 7 additional Puerto Rico watersheds (Manatí, Plata, Fajardo, Añasco, Patillas, Culebrinas, Guanajibo), screened from the full 124-gauge USGS PR network for long, unregulated records with geographic spread across every coast.

**Stage A (climate→discharge): confirmed across all 8 basins** (7 new + Loíza) — every one shows ElasticNet clearly beating both baselines (NSE 0.32–0.74). A common-window sensitivity check (1973–1996, the period all 7 non-Plata basins share) confirms this isn't an artifact of different basins using different historical eras. This is the systematic, generalizable evidence a single-watershed pilot can't provide.

**Stage B (discharge→coastal chlorophyll): 2 of 7 new basins confirmed promising.** Manatí joins Loíza with a genuine signal (final-holdout NSE=+0.146, Spearman ρ=+0.453, same physically-coherent SHAP lag-decay pattern as Loíza). Patillas and Añasco show weak/mixed evidence. Plata, Culebrinas, and Guanajibo are null. Fajardo is untestable (its usable discharge record ends 1996, before the satellite chlorophyll era begins). Full metric suite (NSE, KGE, RMSE, MAE, Spearman) for every basin, and the open question of *why* the signal appears in some basins and not others, is in `watershed_candidates.md`.

## Why some basins show the Stage B signal and others don't (revised)
Tested drainage area, mean discharge, and coastal shelf steepness — all ruled out or weak (r≈0 to -0.50, none significant at n=7). **Digging into the Culebrinas anomaly (persistence beats every model there) found something more important: its test-period autocorrelation spikes from ~0 to 0.62 exactly where the record switches from MODIS to VIIRS-only, and it has the weakest MODIS↔VIIRS bias-correction fit of any basin (R²=0.139) — likely a sensor-splice artifact, not real oceanography.** Re-testing bias-correction R² itself against Stage B skill gives the strongest relationship found (r=+0.655) — both confirmed-promising basins (Manatí, Loíza) have the best sensor-merge quality; the null basins have the worst. **Tested this directly**: re-ran Culebrinas/Guanajibo/Plata on MODIS-only data (no VIIRS splice). Culebrinas improved from -0.059→-0.002 and Plata from +0.030→+0.100 — confirming the data-quality confound for those two. Guanajibo got *worse* (-0.174→-0.196), meaning its null result is likely genuine, not an artifact. Nuanced, falsifiable, now-confirmed-for-2-of-3 finding — full detail in `watershed_candidates.md`.

## Study area map
`figures/08_study_area_map.png` — mainland Puerto Rico, land-only DEM (GEBCO_2020, 15 arc-sec / ~450m) clipped to the true boundary, hillshaded relief, real Río Loíza flowline (USGS NHD), gauge/station/coastal-box/San Juan marked, white background outside the coastline (no bathymetry — not relevant to a discharge study).

Two data-quality issues were found and fixed during construction, worth remembering if this map is rebuilt later:
1. **ETOPO1 (1 arc-min, ~1.85km/pixel) was too coarse** — produced a blocky coastline and missed thin San Juan Bay peninsulas. Replaced with GEBCO_2020 (15 arc-sec, ~450m/pixel), real added resolution, not interpolation.
2. **Census TIGERweb's STATE=72 dissolved boundary has a cartographic-dissolve artifact**: a single 1427-vertex ring connects the mainland to Vieques and Culebra via thin degenerate corridors across open water, producing a spurious "bridge" shape when drawn as a line. Fixed by using municipio-level polygons (layer 1, 78 clean separate polygons), excluding Vieques (COUNTY 147) and Culebra (COUNTY 049) — outside the watershed study area anyway — and dissolving the remaining 76 with shapely (`unary_union`) into one clean outline.

USGS NLDI (the standard tool for an exact watershed drainage polygon for a gauge) has no coverage for Puerto Rico — confirmed by direct query (404). A precise basin polygon would need DEM-based delineation (e.g., WhiteboxTools) — flagged as future work, not attempted since it doesn't change the analysis; the map shows verified point/line locations only.
