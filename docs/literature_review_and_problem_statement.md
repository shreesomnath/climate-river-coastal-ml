# Literature Review & Problem Statement
ASLO 2027 San Juan — Session SS23 (From Droughts to Fisheries: Climate–River–Ocean Linkages Across Timescales)
Author: Somnath Luitel, WKU

## 1. Working title (draft)
"Machine-Learning Assessment of Climate-Driven River-to-Coast Linkages in the Río Grande de Loíza Watershed, Puerto Rico"

## 2. Literature reviewed

| # | Citation (informal) | Scope | Relevant finding | Gap it leaves open |
|---|---|---|---|---|
| L1 | Teleconnections between ENSO and rainfall/drought in Puerto Rico (ResearchGate, ~2018) | Island-wide rainfall/drought vs. ENSO phase | **ENSO is NOT a strong direct driver** of PR monthly/seasonal/annual rainfall; AMO and local factors matter more | Rules out "ENSO → PR drought" as a framing; local precipitation/SPI is the more defensible driver variable |
| L2 | "Impacts of Precipitation Trends on Reservoirs and Rivers in Puerto Rico, 1990–2022" (Sustainability, 2025) | Island-wide precip trends → river/reservoir response (Mann-Kendall, Sen's slope) | Establishes precip → river linkage at trend level | Stops at the river/reservoir; no downstream coastal linkage; trend analysis, not predictive/explainable ML |
| L3 | "Quantifying Effects of Hurricanes Irma and Maria on Coastal Water Quality in PR using VIIRS" (Remote Sensing, 2020) | River discharge → coastal chlorophyll-a / Kd490 via satellite, 2 storm events | Confirms discharge-driven satellite-observable coastal water quality response is real and measurable in PR | Only 2 acute hurricane events; not general climate/drought variability; no ML, no lag/attribution analysis |
| L4 | "Assessment of Coral Reef Eutrophication Thresholds Across PR Using In-Situ and Satellite-Derived Chlorophyll-a" (Estuaries and Coasts, 2025) | Satellite chlorophyll-a tied to reef eutrophication stress | Recent, active interest in chlorophyll-a as an ecological stress proxy in PR | Reef-focused; not framed as a river-to-coast causal chain |
| L5 | Luquillo LTER / Critical Zone Observatory (ongoing) | Long-term streamflow + biogeochemistry, NE mountains | Longest hydrologic records in the Neotropics | Site-specific to Luquillo, not island-scale; not linked to coastal remote sensing |
| L6 | "Forecasting of Macroclimatic Phases Through Stochastic Modeling and ML" (Water, 2025) | ML for regional hydrological/macroclimate phase forecasting (not PR-specific) | RF-based phase classification ~52% accuracy at 3-month lead — modest skill, sets a realistic benchmark expectation | Not applied to PR; not a river-to-coast chain |

## 3. The gap
No existing study chains **drought/precipitation index → river discharge (USGS gauge) → coastal chlorophyll-a response (satellite)** into a single explainable ML pipeline for a Puerto Rico watershed across multiple years. Existing work:
- stops at the river (L2), or
- only looks at the coast during two acute storm events rather than continuous climate variability (L3), or
- already shows ENSO is the wrong driver variable to assume (L1) — a mistake worth avoiding explicitly.

## 4. Problem statement
Puerto Rico's coastal ecosystems are subject to river-borne pulses of freshwater, sediment, and nutrients that are themselves driven by island rainfall variability and drought. While the precipitation→streamflow link and the storm-event streamflow→coastal-water-quality link have each been studied separately in PR, **no study has connected the full chain using a continuous, multi-year record and quantified which lag and which climate variable (local precipitation/SPI, not ENSO) best explains the propagation of drought signal from watershed to coast.** This study uses the Río Grande de Loíza watershed — PR's largest, discharging near San Juan — as a case study, using only freely available USGS, NOAA, and NASA/NOAA satellite data, combined with an explainable ML model (Random Forest/XGBoost + SHAP) to identify the dominant driver and characteristic lag time linking drought to coastal chlorophyll-a response.

## 5. Research questions
1. Does local precipitation/SPI (rather than ENSO) explain the majority of interannual discharge variability in the upper Loíza basin (Caguas gauge, unregulated, 66-yr record)?
2. At what lag does a drought/precipitation anomaly propagate to a detectable chlorophyll-a anomaly in the coastal zone near the Loíza river mouth/San Juan Bay?
3. Does an ML model (RF/XGBoost) meaningfully outperform a naive climatological baseline at predicting coastal chlorophyll-a anomalies from upstream climate + discharge features?

## 6. Known limitation to state up front (not hide)
The lower Loíza basin is regulated by the Carraízo/Loíza reservoir system (gauges below the dam show near-zero, human-controlled flow). Using the Caguas gauge (upstream, unregulated) isolates the natural climate signal, but means the discharge signal reaching the coast is modulated by reservoir operations not captured in this analysis — stated explicitly as a limitation/future-work item (e.g., add reservoir release data if available), not glossed over.
