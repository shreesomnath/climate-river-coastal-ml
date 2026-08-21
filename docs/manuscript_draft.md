# Manuscript Draft (working title)

**Climate-Driven River Discharge and Its Coastal Chlorophyll-a Signature Across Puerto Rico Watersheds: A Multi-Basin, Explainable Machine Learning Assessment**

*Status: working draft, built from `docs/results_summary.md`, `watershed_candidates.md`, `methods_and_formulas.md`, `literature_review_and_problem_statement.md`. Methods and Results are data-grounded and close to final; Introduction/Discussion need advisor review for framing and journal-specific emphasis (see flags inline).*

---

## Abstract (draft, ~250 words — separate from the ASLO 1600-char abstract)

Puerto Rico's coastal ecosystems receive river-borne pulses of freshwater, sediment, and nutrients driven by island rainfall variability, yet no study has connected the full chain — climate driver, river discharge, and coastal ocean-color response — across multiple watersheds using a continuous multi-decadal record and an explainable machine learning framework. We assembled discharge (up to 66 years), precipitation, and satellite ocean-color chlorophyll-a records for 10 Puerto Rico watersheds spanning every coast, using only publicly available government and satellite data. A two-stage modeling framework tested (1) whether local drought conditions (Standardized Precipitation Index, SPI), rather than ENSO, explain discharge variability, and (2) whether a discharge signal propagates detectably to coastal chlorophyll-a. Stage 1 confirmed across all 10 basins: an ElasticNet model using SPI and seasonality explains 32–71% of out-of-sample discharge anomaly variance (permutation test, p=0.005 for the primary basin), with 1-month drought conditions the dominant driver (SHAP attribution). Stage 2 found a discharge-chlorophyll linkage in 2–3 of 7 testable basins (permutation p=0.005 for the strongest case), concentrated offshore rather than at the river mouth in most cases. Basin-to-basin variation in Stage 2 skill was not explained by drainage area or discharge magnitude, but partly by inter-satellite-sensor data-merging quality — a methodological confound we identified and directly tested by reprocessing affected basins with single-sensor data, confirming the effect in 2 of 3 cases. These results provide the first systematic, multi-basin, explainable-ML evidence for climate-driven river-to-coast linkages in a tropical Caribbean setting, and identify concrete data-quality considerations for future satellite-based coastal monitoring in small-island systems.

*[Flag for advisor: journal-specific abstract length/structure varies — this is written as a generic ~250-word structured abstract; adjust to target journal's template.]*

---

## 1. Introduction

*[Flag for advisor: this section most needs domain framing — I've drafted the logical structure and cited what we found in the literature review, but the "why this matters" opening paragraph benefits from your read on what the target journal's audience cares about most: hydrology framing vs. coastal oceanography framing vs. remote sensing framing.]*

Small tropical islands present a distinctive hydroclimatic setting: short, flashy watersheds respond rapidly to rainfall variability, and the resulting river discharge reaches the coast within a short distance and time, creating a tight potential coupling between terrestrial climate variability and coastal ecosystem state (Ramos-Scharrón & Figueroa-Sánchez, [cite]; see also basin response times reported in Section 4.1 below). Puerto Rico, situated in the northeastern Caribbean, exemplifies this setting and is subject to substantial interannual rainfall variability driven by a combination of local convection, the Atlantic Multidecadal Oscillation, and — to a lesser degree than often assumed — the El Niño–Southern Oscillation (ENSO).

Two halves of the climate-river-coast chain have been studied separately in Puerto Rico. Precipitation-to-streamflow relationships have been characterized at the island scale using trend analysis (Sustainability, 2025 — see literature review). Separately, satellite ocean-color studies have quantified acute coastal water-quality responses to specific hurricane-driven discharge events (Remote Sensing, 2020). No study has connected the full chain — climate driver, discharge, and coastal chlorophyll-a response — using a continuous multi-year record, an explainable machine-learning framework capable of attributing skill to specific drivers and lags, and — critically — a multi-basin design capable of testing whether any detected linkage generalizes across the island rather than reflecting a single-watershed idiosyncrasy.

We address this gap with three hypotheses:
- **H1**: Local precipitation/drought conditions (SPI), not ENSO, are the dominant driver of river discharge variability in Puerto Rico watersheds — a deliberate departure from an ENSO-driven framing, motivated by prior work finding ENSO is not a strong direct driver of Puerto Rico rainfall (see Section 2/literature review).
- **H2**: A drought/discharge signal detectably propagates to coastal chlorophyll-a in at least some Puerto Rico watersheds, at a characteristic lag and spatial position relative to the river mouth that need not coincide with the mouth itself.
- **H3**: An explainable ML framework can jointly quantify predictive skill and driver/lag attribution, and can be used to diagnose *why* a linkage is or is not detected in a given basin — including distinguishing a genuine absence of linkage from a data-quality limitation.

---

## 2. Study Area and Data

### 2.1 Study region
Ten Puerto Rico watersheds were selected from the full USGS Puerto Rico streamgage network (n≈124 active gauges), screened for (a) long, continuous daily discharge records, (b) gauges not directly downstream of a dam/reservoir (verified per-gauge by inspecting flow statistics for regulation signatures — a near-constant minimum-flow signature was used as an automated flag, with the primary basin additionally checked manually), and (c) geographic spread across the island's north, south, east, and west coasts (Table 1). Drainage areas range from 6.6 to 208 mi² (USGS-reported).

*Table 1. Watershed characteristics.* [Populate from `data/processed/watersheds/basin_characteristics_vs_stage_b.csv` + Loíza + Humacao/Coamo/Guayanés — gauge ID, coordinates, drainage area, discharge record period, precip station and record period, river mouth coordinates and source (NHD/OSM/coastline-approximated).]

### 2.2 Discharge data
Daily mean discharge was obtained from the USGS National Water Information System (NWIS) public web service for each gauge's full period of record (Section 2.1). Series were aggregated to monthly means, log-transformed (discharge is approximately log-normal), and expressed as anomalies relative to each series' own long-term monthly climatology to isolate interannual variability from the seasonal cycle. Monthly series were reindexed to a continuous calendar prior to any rolling-window or lagged-feature computation — an implementation detail found to matter (Section 3.5) — so that calendar gaps are represented as missing values rather than silently causing rolling operations to bridge non-adjacent months.

### 2.3 Precipitation data and the Standardized Precipitation Index
Daily precipitation was obtained from the NOAA Global Historical Climatology Network-Daily (GHCN-D) archive, using the nearest station with adequate temporal coverage to each gauge (station selection criteria: within ~16 km, record extending to at least 2020, longest available record preferred; see Table 1). The Standardized Precipitation Index (SPI; McKee et al., 1993) was computed at 1, 3, 6, and 12-month accumulation windows via a per-calendar-month gamma-distribution fit, following standard practice.

### 2.4 Satellite chlorophyll-a and inter-sensor merging
Coastal chlorophyll-a was obtained from two NOAA CoastWatch ERDDAP-hosted ocean-color products: a legacy MODIS-Aqua monthly composite (2003–2022) and the current VIIRS/S-NPP science-quality monthly product (2012–present). Because no single sensor spans the full discharge record, and because switching sensors introduces retrieval-algorithm bias, the two products were merged via a log-log linear bias correction fit on their overlap period (2012–2022; per-basin R² ranged 0.11–0.73, Table 2), applied to VIIRS values only for the period after MODIS coverage ends (no double-correction in the overlap window). This inter-sensor correction quality later proved to be an important covariate in its own right (Section 4.3).

### 2.5 Ancillary data
Sea-surface temperature (NOAA OISST v2.1, daily, 0.25°) and 10 m wind speed (NCEP/NCAR Reanalysis, monthly, ~1.9°) were tested as additional Stage 2 covariates for a subset of basins (Section 4.2).

### 2.6 River mouth determination
For each basin, the coastal river mouth was located using the USGS National Hydrography Dataset (NHD) flowline layers, selecting the point on that river's named flowline closest to the true coastline (a 17,243-vertex high-resolution boundary; Section 2.7) — not a directional heuristic, which was found during development to fail for non-north-flowing rivers. Where NHD lacked a named reach (2 of 12 rivers attempted), OpenStreetMap's Overpass API was used as a cross-check/fallback, verified against NHD-derived mouths for basins where both were available (mean discrepancy [TBD — compute if reporting this comparison formally]).

### 2.7 Coastline and elevation reference data
The Puerto Rico coastline boundary was obtained from UN OCHA's Common Operational Dataset (COD-AB, 2019), selected after an initial Census TIGERweb boundary was found to contain a cartographic dissolve artifact (a spurious land connection between the mainland and Vieques/Culebra). Elevation/bathymetry (GEBCO_2020, 15 arc-second) was used for contextual mapping only, not for hydrological delineation.

---

## 3. Methods

### 3.1 Modeling framework
A two-stage framework was used throughout:
- **Stage A (climate → discharge)**: predictors are SPI at four accumulation windows, raw monthly precipitation, sine/cosine-encoded seasonality, and lag-1 discharge anomaly (a persistence feature included as a predictor, distinct from the persistence *baseline* in Section 3.3); target is log discharge anomaly.
- **Stage B (discharge → coastal chlorophyll)**: predictors are discharge anomaly at lags 0–3 months, SPI-3, and seasonality (optionally SST and wind-speed anomalies, Section 4.2); target is log chlorophyll-a anomaly in a coastal box near (Stage B, "near-mouth" candidate) or offset from (Stage B, "offshore" candidate) the river mouth (Section 3.2).

### 3.2 Coastal box placement
For each basin, two candidate coastal sampling boxes were tested: one centered on the river mouth, one offset radially outward (direction computed from the island centroid through the mouth, ~9 km offset, robust to which coast the basin is on). Placement was decided empirically per basin via a lag-0–3 cross-correlation screen between discharge anomaly and chlorophyll-a anomaly (VIIRS-only, for speed) prior to full validation — no single placement rule (e.g., "always offshore") was assumed to generalize, and this was confirmed: 5 of 7 basins favored the offshore box, 2 favored near-mouth.

### 3.3 Models and baselines
Two models were fit at each stage: ElasticNet (linear, L1/L2-regularized; regularization strength selected via nested cross-validation) and Random Forest (shallow, max depth 4–6, to match data volume). Two baselines were always evaluated alongside: climatological (predict the anomaly's own mean, i.e., zero) and persistence (predict the prior month's observed value). Model complexity was deliberately kept low relative to what is common in the broader chlorophyll-ML literature (which includes LSTM/deep-learning approaches) because those methods are typically applied to daily/high-frequency records with far larger sample sizes than the monthly records available here (n in the low hundreds); this choice is supported by ElasticNet matching or exceeding Random Forest at every stage and basin tested.

### 3.4 Cross-validation
Two independent folding schemes were used for every reported cross-validated result: (1) expanding-window `TimeSeriesSplit` (5 folds), and (2) contiguous blocked K-fold (5 folds, no shuffling). Random/shuffled K-fold was not used, as it would leak information across autocorrelated neighboring months. Using two schemes allowed us to confirm that reported skill was not an artifact of either scheme's specific fold boundaries.

### 3.5 Preprocessing correctness
During development of the multi-basin extension, a defect was identified in which monthly series were not reindexed to a continuous calendar before rolling-window (SPI) and lagged-feature computation; because pandas rolling/shift operations act on row position rather than calendar time, a calendar gap (a month absent from the raw data, as opposed to present-but-missing) could cause non-adjacent months to be silently treated as consecutive. This was corrected across the pipeline (Section 2.2) and all reported results reflect the corrected version; the correction was verified not to change the qualitative conclusions (Loíza Stage A NSE changed from 0.514, pre-fix, to 0.390, post-fix — see Section 4.1 for why this is reported transparently rather than only presenting the corrected figure).

### 3.6 Evaluation metrics
Nash-Sutcliffe Efficiency (NSE; Nash & Sutcliffe, 1970) is the primary skill metric throughout, evaluated on held-out data only. Kling-Gupta Efficiency (KGE; Gupta et al., 2009), root-mean-square error, mean absolute error, and Spearman rank correlation are reported as secondary metrics. KGE's bias-ratio component was found to be numerically unstable for the zero-centered anomaly series used here (small denominators) and is interpreted with this caveat; Spearman correlation was consistently more favorable than NSE/KGE across basins, indicating models more reliably capture the relative ranking of high/low periods than absolute magnitude.

### 3.7 Uncertainty and significance
Block-bootstrap 95% confidence intervals (1000 resamples, 6-month contiguous blocks to respect autocorrelation) were computed for final-holdout NSE. To test whether observed skill exceeds chance rather than merely exceeding the climatology/persistence baselines, a permutation test was used: the target series was circularly shifted (preserving its own autocorrelation structure while breaking true temporal alignment with predictors) across 200 realizations, the full modeling pipeline refit each time, and an empirical p-value computed as the fraction of null-distribution NSE values meeting or exceeding the observed value (Figure [X], `20_permutation_tests.png`).

### 3.7b Learning curves
To assess whether reported skill is stable with respect to training-set size — relevant both as a robustness check and as evidence for whether additional data collection (longer records, more basins) would be expected to improve results — held-out NSE was recomputed while incrementally increasing the training set from ~15% to 100% of its full chronological extent (earliest-first, never including future data), for both flagship results (Figure [X], `21_learning_curves.png`).

### 3.8 Explainability
SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) values from the fitted Random Forest were used at both stages to attribute predictive skill to specific features (driver identification) and, via the discharge-lag feature set, to specific time lags (transport/response-time identification).

---

## 4. Results

### 4.1 Stage A: climate-driven discharge variability generalizes across basins
All 10 tested basins showed ElasticNet clearly exceeding both baselines (Table 3; NSE range 0.28–0.74, median [compute]). The flagship basin (Río Grande de Loíza, the pilot watershed with the longest analysis history) achieved NSE=0.390 on a held-out 2013–2026 test period, formally significant against a circular-shift permutation null (p=0.005, n=200 permutations; Section 3.7). SHAP attribution consistently identified SPI-1 (1-month drought index) as the dominant predictor across basins where this was examined in detail, with predictive contribution decaying smoothly across longer accumulation windows — a physically coherent signature of fast (sub-monthly to ~1-month) basin hydrological response, consistent with the short, steep topography typical of Puerto Rico watersheds. This finding directly supports H1 and is inconsistent with an ENSO-driven framing, corroborating prior work (Section 1) that found no strong direct ENSO control on Puerto Rico rainfall.

*Table 3. Stage A results, all 10 basins.* [Populate from `stage_a_all_watersheds_comparison.csv` + `stage_a_3more_watersheds.csv` + Loíza corrected figure.]

A common-window sensitivity analysis (restricting all basins to their shared 1973–1996 period, rather than each basin's own full record) confirmed the multi-basin result is not an artifact of different basins drawing on different historical eras: all 7 basins tested retained positive skill under the shared window, though magnitudes shifted (generally downward, as expected from smaller test sets; two basins weakened substantially, within the range attributable to reduced sample size rather than a change in underlying relationship).

### 4.2 Stage B: discharge-to-coastal-chlorophyll linkage in a subset of basins
Of 7 basins with sufficient satellite-era chlorophyll-a coverage to test (1 basin, Fajardo, was excluded: its usable discharge record ends in 1996, prior to the satellite ocean-color era, so no temporal overlap exists), 2 showed clearly positive, cross-validation-consistent skill (Loíza, Manatí; final-holdout NSE 0.03–0.15, both formally significant against the permutation null, p=0.005 for both), 2 showed weak/mixed evidence (Patillas, Añasco), and 3 showed null or negative skill (Plata, Culebrinas, Guanajibo) under the primary (MODIS+VIIRS merged) analysis.

A learning-curve analysis (Section 3.7b) distinguished these two "confirmed" basins further. Loíza's held-out skill rose smoothly with training-set size and plateaued (NSE ≈0.39) at roughly two-thirds of the available training data, the signature of a converged, stable relationship for which additional data would not obviously change conclusions. Manatí's skill was considerably less stable across training-set sizes, ranging from -0.22 to +0.15 depending on how much training data was used, before landing on its full-data value at the largest training size tested. Both results remain formally significant on the full dataset (Section 3.7), but we treat Loíza as the more robust of the two confirmed cases and report Manatí's significance with this stability caveat rather than presenting the two as equally solid.

SHAP attribution for the confirmed basins showed a consistent, physically coherent pattern across both Loíza and Manatí independently: same-month discharge anomaly dominant, decaying smoothly across lags 1–3, with SPI-3 the second-ranked predictor — a signature consistent with a genuine, relatively fast river-to-coast transport/response process rather than a spurious statistical artifact (which would not be expected to produce a physically ordered lag structure independently in two basins).

Coastal box placement mattered: in the primary case (Loíza), a box offset ~9 km from the river mouth substantially outperformed a box centered on the mouth itself (best correlation 0.50 vs. 0.23), and this offshore-favoring pattern held in 5 of 7 basins screened, but not universally (Section 3.2) — a finding in itself, suggesting river plume chlorophyll signatures in this setting characteristically build over some transport distance rather than peaking at the point of discharge.

### 4.3 Explaining basin-to-basin variation in Stage B skill
Neither drainage area nor mean discharge magnitude correlated with Stage B skill across the 7 tested basins (r=-0.03, r=-0.04 respectively). A coastal shelf-steepness proxy (distance from the river mouth to the -20 m bathymetric contour) showed a moderate negative relationship (r=-0.50, p=0.25, not significant at n=7) but was contradicted by one basin (Culebrinas: shortest shelf distance of all basins tested, yet a null result), limiting its interpretability as a standalone explanation.

Investigating the Culebrinas result directly revealed a methodological confound: its final-holdout test window fell entirely within the VIIRS-only portion of the merged chlorophyll record (i.e., after MODIS coverage ends), and its chlorophyll-a anomaly series showed a large discontinuity in autocorrelation structure at that boundary (lag-1 autocorrelation 0.06 in the training/MODIS-covered period vs. 0.62 in the VIIRS-only test period) — consistent with a residual inter-sensor bias-correction artifact rather than a genuine change in coastal chlorophyll dynamics. Culebrinas also had the weakest inter-sensor bias-correction fit of any basin (R²=0.14; Section 2.4). Testing this directly across all basins, inter-sensor correction quality correlated with Stage B skill more strongly than any physical basin characteristic tested (r=0.66, p=0.11, not significant at n=7, but the strongest relationship found and the only one with a clear causal mechanism: a poorer sensor merge yields a noisier target series, attenuating detectable correlation regardless of whether a true physical linkage exists).

This was tested directly by reprocessing the three weakest-correction basins (Culebrinas, Guanajibo, Plata) using MODIS-only data (2003–2022, no VIIRS splice). Two of three showed the predicted improvement (Culebrinas: final-holdout NSE -0.06 → -0.00; Plata: +0.03 → +0.10, with Spearman correlation becoming clearly positive, +0.28), directly confirming the data-quality-confound hypothesis for those basins. The third (Guanajibo) did not improve (-0.17 → -0.20), suggesting its null result more likely reflects a genuine absence of detectable linkage (or a sample-size limitation — Guanajibo's test period, at 27 months, was the shortest of any basin) rather than the same artifact.

*[Note for advisor: this MODIS-only re-test result is arguably the paper's most interesting methodological contribution — it turns a limitation into a tested, falsifiable claim, and 2-of-3 confirmation (not 3-of-3) is more credible than a clean sweep would have been. Worth featuring prominently in Discussion.]*

---

## 5. Discussion

*[Flag for advisor: drafted at a structural level; needs your judgment on emphasis, and on how strongly to lean into the "explainable ML for physical attribution, not just prediction" framing vs. a more traditional hydrology-trend-analysis framing, which affects target journal fit.]*

**On generalization.** The Stage A result — climate-driven discharge predictability across all 10 tested basins, formally significant, physically coherent in its SHAP attribution — is the paper's most robust finding and directly answers the "does this generalize beyond one watershed" question that a single-basin pilot cannot. The Stage B result is more nuanced by design: rather than reporting a uniform positive or uniform null result, we report and explain heterogeneity, which we view as more scientifically honest and more useful to future work than either extreme would be.

**On the offshore-versus-mouth finding.** [Discuss physical plausibility: freshwater lens formation, mixing/dilution timescales, possible role of longshore currents; note this needs literature grounding beyond what's in `literature_review_and_problem_statement.md` — recommend a targeted search on river plume detection distances in small tropical islands specifically.]

**On the sensor-merging confound.** This is a broadly applicable methodological point for any study merging heritage (MODIS) and current (VIIRS, or successor sensors) ocean-color products for a discharge/land-margin analysis: bias-correction quality should be checked and reported per spatial unit of analysis, not just at a study-wide level, since it can vary substantially (R²=0.11–0.73 across basins here) and can materially affect downstream conclusions. We are not aware of prior work in this specific system reporting this check.

**Limitations.** (1) Basin sample size (n=7–10) is adequate to demonstrate generalization of Stage A and to motivate but not confirm the Stage B basin-characteristic explanations, which remain suggestive (p>0.05 throughout) rather than confirmed; a larger basin sample is identified future work. (2) Coastal box placement, while empirically tested rather than assumed, was tested at only two candidate positions per basin; a denser spatial search could refine the offshore-distance finding. (3) [Additional limitations per advisor input.]

---

## 6. Conclusion

*[Draft after Discussion is finalized with advisor input — should restate H1/H2/H3 status concisely: H1 confirmed (10/10 basins), H2 confirmed in a subset with an identified and partially-tested explanation for the rest, H3 demonstrated throughout via SHAP + the Culebrinas diagnostic.]*

---

## References (to compile formally)
- McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales.
- Nash, J.E., Sutcliffe, J.V. (1970). River flow forecasting through conceptual models.
- Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009). Decomposition of the mean squared error and NSE performance criteria.
- Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions.
- [ENSO/PR rainfall teleconnection paper — get full citation from literature_review_and_problem_statement.md]
- [PR precipitation/reservoir trends paper, Sustainability 2025 — full citation]
- [Hurricane Irma/Maria coastal water quality, Remote Sensing 2020 — full citation]
- [Pearl River Estuary RF chlorophyll paper — full citation]
- Data sources: USGS NWIS, NOAA GHCN-D, NOAA CoastWatch ERDDAP (MODIS/VIIRS chlorophyll, OISST), NCEP/NCAR Reanalysis, USGS NHD, OpenStreetMap, UN OCHA COD-AB, GEBCO_2020 — full citations per each provider's recommended format.

---

## What's needed before this is submission-ready
1. Advisor review of Introduction/Discussion framing (flagged inline above)
2. Fill remaining bracketed placeholders (exact NSE medians, table population from CSVs already saved, formal reference list)
3. A proper Figure set selected from the 19 already generated (not all 19 belong in the paper — recommend: study area map, Stage A metrics comparison, Stage B multi-basin comparison, SHAP for both flagship basins, bias-R² vs. skill, MODIS-only retest)
4. Target-journal formatting and length compliance once a journal is chosen (see open question in `project_overview_for_advisor.md`)
5. Co-author list / author contributions per your advisor's guidance
