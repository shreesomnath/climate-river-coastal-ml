"""
Merge MODIS-Aqua (2003-01 to 2022-03) and VIIRS S-NPP (2012-01 to 2026-06)
chlorophyll-a into one continuous, bias-corrected series -- same principle
merged ocean-color products like OC-CCI use: fit an inter-sensor
correction on the overlap period, then splice.

Rule: use MODIS directly where available (2003-01 to 2022-03, the longer-
running, more established product); use VIIRS corrected onto the MODIS
scale for months after MODIS ends (2022-04 onward). No double-adjustment
in the overlap window.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

modis = pd.read_csv(RAW / "loiza_mouth_chlorophyll_monthly.csv")
viirs = pd.read_csv(RAW / "loiza_mouth_chlorophyll_viirs_monthly.csv")
modis["log_chl"] = np.log(modis["chl_a_mg_m3_mean"])
viirs["log_chl"] = np.log(viirs["chl_a_mg_m3_mean"])

overlap = modis[["year_month", "log_chl"]].merge(
    viirs[["year_month", "log_chl"]], on="year_month", suffixes=("_modis", "_viirs")
)
reg = LinearRegression().fit(overlap[["log_chl_viirs"]], overlap["log_chl_modis"])
r2 = reg.score(overlap[["log_chl_viirs"]], overlap["log_chl_modis"])
print(f"Inter-sensor bias correction (log-log, n={len(overlap)} overlap months): "
      f"log(MODIS) = {reg.intercept_:.3f} + {reg.coef_[0]:.3f} * log(VIIRS), R^2={r2:.3f}")

viirs["log_chl_corrected"] = reg.predict(viirs[["log_chl"]].rename(columns={"log_chl": "log_chl_viirs"}))

modis_part = modis[["year_month", "log_chl"]].rename(columns={"log_chl": "log_chl_merged"})
modis_part["source"] = "MODIS"

viirs_post_modis = viirs[viirs["year_month"] > modis["year_month"].max()][
    ["year_month", "log_chl_corrected"]
].rename(columns={"log_chl_corrected": "log_chl_merged"})
viirs_post_modis["source"] = "VIIRS_corrected"

merged = pd.concat([modis_part, viirs_post_modis], ignore_index=True).sort_values("year_month")
merged["chl_a_mg_m3_merged"] = np.exp(merged["log_chl_merged"])
merged = merged.reset_index(drop=True)

OUT = PROC / "loiza_mouth_chlorophyll_merged_monthly.csv"
merged.to_csv(OUT, index=False)
print(f"Merged series: {len(merged)} months, {merged['year_month'].min()} to {merged['year_month'].max()}")
print(f"  MODIS-native: {(merged['source']=='MODIS').sum()} months")
print(f"  VIIRS-corrected: {(merged['source']=='VIIRS_corrected').sum()} months")
print(f"Saved to {OUT}")
