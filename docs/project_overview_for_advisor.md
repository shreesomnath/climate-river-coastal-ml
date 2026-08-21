# Project Overview (for advisor discussion)

## The one-sentence version
We're testing whether drought/rainfall variability in a Puerto Rico watershed propagates — through river discharge — to a measurable change in coastal ocean chlorophyll, using only free satellite/government data and machine learning, with the goal of an ASLO 2027 abstract now and a full journal paper later.

## Problem statement
Puerto Rico's coastal ecosystems receive river-borne pulses of freshwater, sediment, and nutrients that are themselves driven by island rainfall variability and drought. Two pieces of this chain have been studied separately before (rainfall→river trends; river→coast during specific hurricanes), but nobody has connected the **full chain — climate driver → river discharge → coastal chlorophyll response — using a continuous multi-year record with an explainable ML model** that identifies which climate variable and what time lag actually explains the linkage. That's the gap.

## Hypothesis
1. **H1**: Local precipitation/drought (SPI), not ENSO, is the dominant driver of river discharge variability in this basin. *(We deliberately did NOT assume ENSO — a literature check found ENSO isn't a strong direct PR rainfall driver, so testing SPI instead was itself a considered methodological choice.)*
2. **H2**: A drought/discharge signal detectably propagates to coastal chlorophyll-a, at some characteristic lag and in some spatial zone relative to the river mouth (not necessarily *at* the mouth).
3. **H3**: An explainable ML model (not a black box) can quantify both the skill of that prediction and which specific driver/lag matters — usable for attribution, not just forecasting.

## What we actually did (plain-language version)
- Picked Río Grande de Loíza (PR's largest watershed, discharges near San Juan) as the case study, using an upstream USGS gauge with a 66-year record that's *not* affected by the downstream reservoir (so we're measuring nature, not dam operators).
- Pulled real precipitation records and computed the standard drought index (SPI) hydrologists use.
- **Result on H1**: confirmed. A simple model using SPI + seasonality explains ~39% of the year-to-year variation in river discharge (revised down from an initial ~51% after fixing a real data-preprocessing bug — a calendar-gap issue in the precipitation record that a multi-watershed sanity check exposed; still a genuinely strong, clearly-above-baseline result), and short-term (1-month) drought conditions matter most — the basin reacts fast, within about a month.
- Pulled 20+ years of satellite ocean-color data (chlorophyll-a, a proxy for coastal water quality/productivity) near the coast.
- **Result on H2**: initially null (no linkage detected) — but that first attempt used a coastal sampling box that, in hindsight, wasn't well positioned relative to the true river mouth. After relocating the box based on the actual mapped river channel (not a guess), we found a real, modest, still-being-confirmed linkage: about 12-15% of coastal chlorophyll variation is explained by upstream discharge, consistently across two independent statistical validation methods. It's not yet a slam-dunk (the confidence interval still touches zero), but it's a genuine, promising, physically-sensible finding — the effect is strongest at the same time the river discharges (not delayed), and gets weaker the further back you look, exactly as you'd expect if it's a real transport signal.
- **Result on H3**: yes — used SHAP (a standard explainability method) throughout, so every result comes with "and here's specifically what's driving it," not just a prediction number.

## Where this stands right now (2026-08-20)
- Stage 1 (climate → river): strong, validated, ready to write up.
- Stage 2 (river → coast): promising lead, currently adding more predictors (sea-surface temperature helped; wind speed helped a little) to firm it up before claiming it as a confirmed result.
- Everything — every dataset, every script, every number — is saved and reproducible in this project folder (not just numbers in a chat transcript).

## The abstract vs. the paper — two different bars
- **ASLO abstract (due Sept 15, 26-day timeline)**: this is enough. A strong Stage 1 result plus an honest, hedged "promising Stage 2 finding, refinement ongoing" is a completely normal, defensible abstract.
- **Journal publication (the actual goal)**: a single watershed is a solid pilot but reviewers will ask "does this generalize?" The realistic path there is scaling the exact same pipeline (already built, already parameterized by gauge/box) across multiple Puerto Rico watersheds to turn this into a systematic, generalizable study rather than a single case study — discussed as a next phase, timeline permitting between the abstract deadline and the Feb/March 2027 meeting.

## What I'd want feedback on from an advisor
1. Is the Stage 2 "promising but not confirmed" framing the right call for the abstract, or should we wait for a fully confirmed result before submitting anything about coastal linkage?
2. How many additional watersheds would be scientifically convincing for a journal submission — 3? 5? all ~124 PR gauges filtered down to unregulated/long-record ones?
3. Any preference on target journal (affects how much additional rigor/scope to build in now vs. later)?
