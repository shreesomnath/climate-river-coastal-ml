"""
Stage B preprocessing: Discharge (+climate) -> coastal chlorophyll-a.
Uses the bias-corrected MODIS+VIIRS merged record (2003-2026, see
09_merge_chlorophyll_sensors.py) joined against Stage A's discharge/SPI.
Outputs data/processed/stage_b_monthly.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

stage_a = pd.read_csv(PROC / "stage_a_monthly.csv")
stage_a["ym"] = pd.PeriodIndex(stage_a["ym"], freq="M")

chl = pd.read_csv(PROC / "loiza_mouth_chlorophyll_merged_monthly.csv")
chl["ym"] = pd.PeriodIndex(chl["year_month"], freq="M")
clim = chl.groupby(chl["ym"].dt.month)["log_chl_merged"].transform("mean")
chl["log_chl_anomaly"] = chl["log_chl_merged"] - clim

df = stage_a.merge(chl[["ym", "log_chl_anomaly", "source"]], on="ym", how="inner")
df = df.sort_values("ym").reset_index(drop=True)

# Discharge anomaly at lags 0-3 months (river signal arriving at the coast with delay)
for lag in (0, 1, 2, 3):
    df[f"log_q_anomaly_lag{lag}"] = df["log_q_anomaly"].shift(lag) if lag > 0 else df["log_q_anomaly"]

df.to_csv(PROC / "stage_b_monthly.csv", index=False)
print(f"Stage B monthly dataset: {len(df)} rows, {df['ym'].min()} to {df['ym'].max()}")
print(df["source"].value_counts().to_string())
