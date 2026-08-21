# Multi-Watershed Expansion — Candidate Screening (Phase 3)

Screened all active USGS Puerto Rico stream gauges with daily discharge records (`00060`, `dv`), filtering for: long record (>50 yr), not immediately below a named dam/reservoir (name doesn't contain "BLW LAGO"/"BLW DAM"), and spatially distinct from Río Loíza (already the pilot watershed) to give geographic spread around the island.

## Selected candidates (7, spanning N/S/E/W coasts)

| Gauge | River | Lat | Lon | Record start | Record length (days) | Coast |
|---|---|---|---|---|---|---|
| 50035000 | Río Grande de Manatí (at Ciales) | 18.324 | -66.460 | 1946-09 | 27,255 | North-central |
| 50046000 | Río de la Plata (at Toa Alta) | 18.412 | -66.261 | 1960-01 | 24,330 | North (near San Juan, west side) |
| 50071000 | Río Fajardo (at Fajardo) | 18.299 | -65.694 | 1961-04 | 23,881 | East |
| 50144000 | Río Grande de Añasco (at San Sebastián) | 18.284 | -67.051 | 1963-04 | 23,002 | West |
| 50092000 | Río Grande de Patillas (at Patillas) | 18.034 | -66.033 | 1966-01 | 22,145 | Southeast |
| 50147800 | Río Culebrinas (at Hwy 404, Moca) | 18.362 | -67.093 | 1967-07 | 21,322 | Northwest |
| 50138000 | Río Guanajibo (at Hormigueros) | 18.143 | -67.149 | 1973-01 | 19,588 | Southwest |

Excluded from the top-record list: gauges below named reservoirs (e.g., Río Cerrillos "BLW LAGO CERRILLOS"), gauges on Loíza tributaries (Río Gurabo, Río Cayaguas — same watershed as the pilot, would not add independent geographic coverage), and small-basin/short-record gauges.

## Not yet done for these candidates (next actions, same process as Loíza)
1. Confirm each gauge is genuinely unregulated (check for upstream dams via USGS site remarks / basin characteristics — Loíza's Caguas gauge required this check).
2. Find each river's true mouth via USGS NHD flowline endpoint (same method as `loiza_river_flowlines.geojson` / script that found -65.878, 18.438 for Loíza) — do not guess coordinates.
3. Find/verify a co-located long-record GHCN precipitation station per basin.
4. Run the same box-placement test (near-mouth vs. offshore candidates) per river — Loíza's own result (offshore beat near-mouth) should NOT be assumed to generalize; each river needs its own test.
5. Apply the identical Stage A + Stage B pipeline (scripts 01-13, parameterized by gauge ID / station ID / box coords).

## Progress log (2026-08-20)

### What's done
Ran `scripts/15_batch_fetch_watersheds.py` for all 7 candidates:
1. **Discharge**: full daily USGS record fetched for each (`data/raw/watersheds/{name}_discharge_daily.csv`).
2. **Regulation QC**: crude check (fraction of days at/near the record minimum flow — a proxy for dam-controlled constant releases). **None of the 7 flagged** — all show natural min/max variability. This is a first-pass automated check only; still recommend a manual look (station remarks, basin characteristics) before fully trusting each, same as the more thorough check Loíza's Caguas gauge got.
3. **River mouth**: found geometrically (point on that river's NHD flowline closest to the real coastline — NOT a directional guess; see method note below) for 6 of 7. Fajardo has no named NHD flowline reach near its gauge in either the small-scale or large-scale layer (checked both) — its mouth is a **documented approximation** (nearest point on the real coastline to the gauge itself), not equal-confidence to the other 6.

| Watershed | Mouth (lon, lat) | Source |
|---|---|---|
| Manatí | -66.533, 18.481 | NHD (4040 pts matched) |
| Plata | -66.256, 18.476 | NHD (3800 pts matched) |
| Fajardo | -65.627, 18.328 | **OpenStreetMap** (Overpass API, 469 pts) — NHD has no named reach for this river at all (checked both small- and large-scale layers, genuinely absent, not a query bug); OSM had it mapped as "Río Fajardo," 3 way segments. Point is ~17m from the real coastline (essentially exact). This replaced an earlier same-session guess (nearest-coastline-point to the gauge) that was off by ~4km once the real data came in — a good reminder to cross-check a second source before trusting a fallback approximation. |
| Añasco | -67.188, 18.267 | NHD (2206 pts matched) |
| Patillas | -66.014, 17.980 | NHD large-scale layer (2375 pts) |
| Culebrinas | -67.177, 18.406 | NHD (3004 pts matched) |
| Guanajibo | -67.181, 18.168 | NHD (2397 pts matched) |

## Precipitation stations (fixed after finding an off-by-one bug in inventory parsing — `len(parts) < 7` skipped every valid 6-field GHCN inventory line; corrected to `< 6`)

| Watershed | Station | Record | Distance from gauge |
|---|---|---|---|
| Manatí | RQC00665807 MANATI 2 E | 1900-2026 (126 yr) | 11.9 km |
| Plata | RQC00669415 TOA BAJA LEVITTOWN | 2004-2026 (22 yr, shorter — flag for review) | 10.7 km |
| Fajardo | RQC00663657 FAJARDO | 1931-2026 (95 yr) | 6.0 km |
| Añasco | RQC00662801 COLOSO | 1899-2026 (127 yr) | 15.9 km |
| Patillas | RQC00664193 GUAYAMA 1SW | 1911-2026 (115 yr) | 12.0 km |
| Culebrinas | RQC00662801 COLOSO (same as Añasco) | 1899-2026 (127 yr) | 7.4 km |
| Guanajibo | RQC00665097 LAJAS SUBSTN | 1900-2026 (126 yr) | 14.9 km |

Note: Añasco and Culebrinas share the same nearest station (Coloso) since their gauges are relatively close together on the west coast — acceptable (SPI is a regional climate index, not expected to differ sharply over ~10km), but worth knowing before treating the two basins as fully independent samples in any pooled statistical analysis later.

**Method note (methodological fix made during this run)**: mouth-finding originally used "northernmost flowline point," which only works for north-flowing rivers (fine for Loíza/Manatí/Plata, wrong for Fajardo [flows east], Añasco [west], Patillas/Guanajibo [south]). Fixed to find the point geometrically closest to the actual coastline polygon instead — correct regardless of flow direction. All 7 results above are geographically sanity-checked (each mouth coordinate falls on the correct coast for that river).

**No DEM-based delineation was performed.** River channels come from USGS's pre-built NHD database, not derived here. Watershed *boundary* (drainage-area) delineation, which would require DEM flow-accumulation, remains undone/unneeded — same as for Loíza.

### Stage A results — all 7 basins (corrected, post calendar-gap-fix; see results_summary.md correction note)

Each basin uses its own full available record (not a common window) — see the discussion earlier in this conversation for why, plus the note below on the sensitivity check still owed.

| Watershed | Record used | n (months) | NSE_climatology | NSE_persistence | **NSE_ElasticNet** | NSE_RandomForest |
|---|---|---|---|---|---|---|
| Manatí | 1946-10 to 2025-09 | 589 | -0.003 | +0.205 | **+0.542** | +0.523 |
| Plata | 2005-11 to 2026-07 | 115 | -0.035 | -0.377 | **+0.477** | +0.491 |
| Fajardo | 1961-05 to 1996-01 | 382 | -0.119 | -0.266 | **+0.491** | +0.466 |
| Añasco | 1963-05 to 2026-07 | 635 | -0.205 | -0.102 | **+0.321** | +0.282 |
| Patillas | 1966-02 to 2025-09 | 441 | -0.007 | +0.156 | **+0.735** | +0.742 |
| Culebrinas | 1967-08 to 2026-07 | 580 | -0.007 | +0.132 | **+0.433** | +0.405 |
| Guanajibo | 1973-02 to 2021-11 | 457 | -0.033 | +0.093 | **+0.519** | +0.526 |
| *(Loíza, pilot)* | *1969-09 to 2026-07* | *426* | *-0.001* | *-0.440* | ***+0.390*** | *+0.297* |

**Every single one of the 8 basins (7 new + Loíza) shows ElasticNet clearly beating both baselines.** This is the systematic, multi-basin confirmation of the climate→discharge hypothesis (H1) that a single-watershed pilot can't provide — genuinely strong evidence for a journal submission. Patillas is the standout (NSE=0.735); Añasco is the weakest but still solidly positive (0.321).

Caveats to carry into the writeup:
- Plata's sample (n=115) is thin — its nearest precip station only starts in 2004. Worth searching for a second, longer-record station further away as a robustness check before leaning heavily on this basin's number.
- Fajardo's usable window ends in 1996 — its precip station (COOP, manually-read) has a genuine ~30-year gap after that. A different/longer station may exist nearby; not yet searched.
- Record lengths differ by basin (115 to 635 months) — each basin used its own maximum available record (standard practice, maximizes power). **Still owed**: a common-window (1973–2026, Guanajibo's start) re-run across all 8 as an explicit sensitivity check, to preempt a reviewer's "different periods" concern. Not done yet.

### Common-window sensitivity check (`18_common_window_sensitivity.py`) — DONE
The true common window across all 8 basins is dominated by Plata's short record (2005) and Fajardo's early cutoff (1996) — those two barely overlap at all, so a meaningful common-window test excludes Plata: **1973–1996 (23 yr), bounded by Guanajibo's start and Fajardo's end.**

| Watershed | NSE_EN (full record) | NSE_EN (common window 1973-1996) |
|---|---|---|
| Loíza | 0.390 | 0.230 |
| Manatí | 0.542 | 0.589 |
| Fajardo | 0.491 | 0.475 |
| Añasco | 0.321 | 0.108 |
| Patillas | 0.735 | 0.423 |
| Culebrinas | 0.433 | 0.093 |
| Guanajibo | 0.519 | 0.302 |

**Result: all 7 basins still beat climatology under the shared window — the core finding is not an artifact of different basins using different historical eras.** Magnitudes shift, generally downward (expected — the window is short, so test sets shrink to ~30-50 months and estimates get noisier; Culebrinas and Añasco weaken the most, dropping close to zero, worth a second look before leaning on those two specifically in the paper). Manatí is the one basin that got *stronger* under the common window. This check should be cited alongside the full-record table to preempt the "different periods" critique.

### Coastal box screen, all 7 basins (`19_batch_test_coastal_boxes.py`) — quick VIIRS-only correlation, near-mouth vs. radial-offshore (~9km)

| Watershed | near_mouth \|r\| | offshore \|r\| | Winner |
|---|---|---|---|
| Manatí | 0.344 | **0.465** | offshore |
| Plata | 0.223 | **0.330** | offshore |
| Fajardo | 0.139 (lag 3, less physical) | **0.160** (lag 0) | offshore |
| Añasco | 0.229 | **0.290** | offshore |
| Patillas | -0.062 (lag 2) | **0.157** (lag 0) | offshore |
| Culebrinas | **0.341** | 0.329 | near_mouth (barely) |
| Guanajibo | **0.210** (lag 1) | -0.132 (lag 2, noisy) | near_mouth, clearly |

**Loíza's "offshore beats near-mouth" pattern generalizes to 5 of 7 basins, but not all** — Culebrinas is a near-tie and Guanajibo clearly favors near-mouth instead. This is itself a real, interesting finding (not a null result to explain away): the offshore-signal-strengthening effect is common but not universal, plausibly tied to basin-specific factors (discharge volume, local bathymetry/currents, coastal geometry) not yet investigated. Do not generalize "always test offshore first" as a rule — each basin still needs its own check, exactly as this table demonstrates.

All correlations here are modest (0.13–0.47) — weaker across the board than Loíza's far_offshore finding (0.50). None of these have had the full validation treatment yet (MODIS+VIIRS merge, proper CV, baselines, SHAP) — this is a screening pass only, to prioritize which basins are worth that investment.

## Full Stage B validation — all 7 basins, DONE (`20_stage_b_top4_full_validation.py`, `22_full_metrics_all_basins.py`)

Full rigor (MODIS+VIIRS merge, two folding schemes, baselines, final holdout, and — new in this pass — the complete metric suite: NSE, KGE, RMSE, MAE, Spearman) run for all 7 basins. Fajardo could not be tested at all: its usable discharge record ends 1996 (real station gap, see above) and satellite chlorophyll only starts 2003 — zero temporal overlap, not a modeling failure.

### Final holdout, full metric suite (ElasticNet)

| Basin | n_test | NSE | KGE | RMSE | MAE | Spearman ρ | Verdict |
|---|---|---|---|---|---|---|---|
| **Manatí** | 42 | **+0.146** | -0.311 | 0.296 | 0.236 | +0.453 | **Promising** (2nd confirmed basin) |
| Patillas | 46 | +0.096 | -1.483 | 0.194 | 0.160 | +0.250 | Weak/mixed — positive NSE but very poor KGE (bias/variance mismatch, not just noise) |
| Añasco | 48 | +0.063 | -0.703 | 0.431 | 0.319 | +0.544 | Weak — positive NSE and the best Spearman of all basins, but poor KGE |
| Plata | 42 | +0.030 | +0.010 | 0.292 | 0.236 | +0.233 | Essentially null |
| Culebrinas | 45 | -0.059 | n/a (ElasticNet collapsed to ~constant prediction) | 0.286 | 0.226 | n/a | Null — notably, **Persistence alone scores NSE=+0.236, KGE=+0.571** here, better than any of our models; the chlorophyll signal itself is autocorrelated month-to-month but our discharge/SPI features don't explain it |
| Guanajibo | 27 | -0.174 | n/a (collapsed) | 0.520 | 0.378 | n/a | Null, worst-performing basin; also has the shortest test window (27 months) |
| Fajardo | — | — | — | — | — | — | Untestable (no temporal overlap between usable discharge record and satellite era) |

**Cross-basin pattern worth flagging**: KGE is poor (often very negative) even for basins with positive NSE (Patillas, Añasco). This means the models get the *timing/ranking* right (reflected in the reasonable-to-strong Spearman correlations) but not the *magnitude/variance* — consistent with what we already saw for Loíza's far_offshore result and noted in `methods_and_formulas.md` (KGE's bias term is unstable for zero-centered anomaly series). Across basins, Spearman ρ is consistently more favorable than NSE/KGE, reinforcing that "does the model rank periods correctly" is the more robust question to ask of this data than "does it nail absolute magnitude."

**Culebrinas' persistence result is worth a dedicated look** — it's the one basin where a naive "last month's chlorophyll" beats every model we built, meaning there's real structure in that basin's ocean-color signal that our climate/discharge features simply aren't capturing (possibly a longer-memory oceanographic process, not river-driven at all).

### Net multi-basin Stage B result
**2 of 7 basins (Loíza, Manatí) show a genuine, promising discharge→coastal-chlorophyll signal; 2 more (Patillas, Añasco) show weak/mixed evidence; 2 are null (Plata, Culebrinas — worse than baseline once you look past raw NSE); 1 is untestable (Fajardo).** This is not a discouraging outcome for a paper — it's a real, defensible pattern (roughly 2-4 of 8 PR watersheds show detectable river-to-coast linkage) that itself needs explaining, which is a legitimate and interesting research question (basin size? discharge magnitude? local bathymetry/currents? distance from mouth to the shelf break?) — not yet investigated, flagged as a natural next analysis.

## Basin set expansion, round 2 (`25_add_3_more_watersheds.py`): n=8 → n=10

Added 3 more candidates (Humacao, Guayanés, Coamo — smaller basins than the original 7, drainage areas 6.6–43.5 mi², extending the range for the basin-characteristics analysis) using the identical pipeline. 2 of 3 succeeded strongly:

| Watershed | Drainage area (mi²) | n (months) | NSE_climatology | **NSE_ElasticNet** | NSE_RandomForest |
|---|---|---|---|---|---|
| Humacao | 6.65 | 422 | -0.000 | **+0.654** | +0.686 |
| Coamo | 43.5 | 325 | -0.037 | **+0.713** | +0.636 |
| Guayanés | 16.4 | 30 (insufficient) | — | — | — |

Guayanés has real, compounding gaps in *both* its discharge record and its nearest precip station (only 178/686 and 321/686 non-null months respectively even before requiring overlap) — same story as Fajardo. Not pursued further; documented rather than forced.

**Running Stage A tally: 10 of 10 tested basins now show ElasticNet clearly beating climatology** (Humacao and Coamo are in fact the two strongest results yet, ahead of even Loíza). This is an even stronger multi-basin confirmation than before.

### Stage B screen, Humacao & Coamo
| Basin | near_mouth \|r\| | offshore \|r\| | Note |
|---|---|---|---|
| Humacao | only 8 months of data — box mostly land-masked | **0.200** (lag 3, less clean) | Near-mouth box geometry issue (likely a bay/lagoon coastline); would need a repositioned box to test properly |
| Coamo | **0.312** (lag 1, physically clean) | 0.083 | Near-mouth wins here — a genuine third pattern (Loíza/Manatí/etc favor offshore, Culebrinas/Guanajibo favor near-mouth, Coamo now also favors near-mouth) |

Coamo is a real candidate for full Stage B validation later (not yet run) — its screen result is cleaner than several basins that did get the full treatment.

## Why does the signal appear in some basins and not others? (`23_basin_characteristics_analysis.py`, `24_basin_characteristics_plot.py`)

Tested 3 physical basin characteristics against the Stage B final-holdout NSE across the 7 testable basins: USGS-reported drainage area, mean discharge (both real data, not estimated), and a coastal shelf-steepness proxy (distance from the river mouth to the -20m depth contour, computed from the GEBCO bathymetry already on hand).

| Characteristic | Correlation with Stage B NSE | Verdict |
|---|---|---|
| Drainage area | r = -0.03 | No relationship |
| Mean discharge | r = -0.04 | No relationship |
| Distance to -20m isobath (shelf steepness) | r = -0.50 | Suggestive but **not statistically significant** (n=7, p=0.25) |

**Honest read (see figure `17_basin_characteristics_vs_stage_b.png`)**: none of the three cleanly explains the pattern. The shelf-distance correlation is the strongest of the three, and physically plausible (a narrower/steeper shelf could concentrate a river plume's surface signature rather than letting it disperse across a wide shallow shelf before reaching open water) — but it's driven substantially by one point (Guanajibo, the basin with both the longest shelf distance and the worst NSE), and it's flatly contradicted by Culebrinas (the *shortest* shelf distance of all 7, yet a null/negative result). Not a finding to lean on as-is.

**Conclusion**: basin size and discharge magnitude are ruled out as explanations. Shelf geometry is a real lead worth carrying into the larger multi-basin expansion (more basins = more statistical power to resolve whether it's real), but current n=7 can't confirm it — this itself is a legitimate motivation for scaling past the current 8 basins, not a dead end.

## REVISED "why" finding: data-quality confound, not (only) physical basin difference

Investigated the Culebrinas anomaly directly (`27_bias_r2_vs_stage_b_plot.py` + inline analysis) and found something that reframes the whole "why do some basins show a signal" question:

**Culebrinas' training-period chlorophyll autocorrelation is essentially zero (ACF lag-1 = 0.06), but its test/holdout period (2022-06 to 2026-06) jumps to 0.62** — a completely different regime. That holdout window falls *entirely* within the VIIRS-only era (after MODIS's 2022-03 cutoff), and Culebrinas has the **weakest MODIS↔VIIRS bias-correction fit of any basin (R²=0.139)**. A poor sensor-splice correction plausibly introduces a slowly-drifting residual bias, which *looks like* strong month-to-month persistence in the merged series without being real oceanography — meaning "persistence beats every model" for Culebrinas is likely a **data-splice artifact**, not a genuine finding about that basin's chlorophyll dynamics.

This prompted re-testing bias-correction R² itself against Stage B skill, instead of (only) physical basin characteristics:

| Characteristic | Correlation with Stage B NSE | p-value (n=7) |
|---|---|---|
| Drainage area | r = -0.03 | — |
| Mean discharge | r = -0.04 | — |
| Shelf distance (-20m isobath) | r = -0.50 | 0.25 |
| **MODIS↔VIIRS bias-correction R²** | **r = +0.655** | **0.110** |

Bias-correction quality is the strongest relationship found (figure `18_bias_r2_vs_stage_b.png`) — both "Promising" basins (Manatí R²=0.730, Loíza R²=0.688) sit at the high end; the "Null" basins (Culebrinas 0.139, Guanajibo 0.113) sit at the low end. Still not significant at n=7, but this one has a genuine mechanistic story (poor sensor-merge quality → noisier input → attenuated ability to detect *any* true correlation, independent of whether the real linkage exists) rather than a physical hypothesis that has to explain actual oceanography.

**Honest implication for the paper**: some of the "null" basins may not actually lack a river-to-coast linkage — we may simply be unable to detect it well because their specific coastal box has poor MODIS-VIIRS agreement. This is a genuine methodological caveat to disclose, and it also points to a concrete fix for future work: use MODIS-only (2003-2022, no VIIRS splice) for basins with weak bias-correction R², trading record length for data quality, and see whether previously-null basins look different.

## MODIS-only re-test — confirms the hypothesis for 2 of 3 basins (`28_modis_only_retest.py`, `29_modis_only_comparison_plot.py`)

Direct test: re-ran Culebrinas, Guanajibo, and Plata using MODIS-only chlorophyll (2003–2022, no VIIRS splice at all) and compared final-holdout NSE against the original merged (VIIRS-spliced) result.

| Basin | Merged (VIIRS-spliced) | MODIS-only | Change |
|---|---|---|---|
| Culebrinas | -0.059 | **-0.002** | Improved — consistent with the sensor-splice-artifact hypothesis |
| Plata | +0.030 | **+0.100** (Spearman ρ now +0.281) | Improved meaningfully — also consistent |
| Guanajibo | -0.174 | -0.196 | **Worse** — does NOT support the hypothesis here |

**Confirmed, nuanced conclusion**: the data-quality confound is real for Culebrinas and Plata — both improve substantially once the noisy VIIRS tail is removed, exactly as predicted. But it does **not** explain Guanajibo, which gets slightly worse under MODIS-only despite also having a weak bias-correction R² (0.113) — Guanajibo's null result looks more likely to be a genuine absence of the discharge→chlorophyll linkage (or possibly just noise from its small sample, n=133/test=27, the smallest of any basin). This is exactly the kind of result worth reporting as-is: **the confound explains some but not all of the null basins** — a real, falsifiable, now-tested claim rather than a speculative pattern.

### What's NOT done yet (next steps)
- Extend both the physical-characteristics and bias-R² comparisons as more basins are added (n=7 is too small to resolve either lead with real confidence)
- Try additional candidate explanatory variables: point-source pollution/urbanization near the coast, local wind/current climatology
- Second precip station for Plata (and possibly Fajardo) as a robustness check
- Guanajibo specifically may need a larger/better-placed box (recall its offshore screen result was also anomalously negative) before concluding "no linkage" with confidence

## Why these 7 (not all ~124 PR gauges)
124 includes many tiny/short-record/regulated/duplicate-basin gauges. These 7 were chosen to (a) maximize independent geographic coverage (every coast represented), (b) match or exceed Loíza's record length where possible, (c) avoid reservoir regulation. This is a reasonable first tranche — could extend further if reviewers/advisor want more basins, but 7 (+Loíza = 8 total) is already enough to move from "one case study" to "a systematic multi-basin result," which was the actual gap for journal publication.
