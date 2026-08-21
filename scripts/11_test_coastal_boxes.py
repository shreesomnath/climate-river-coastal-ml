"""
Stage B refinement: test alternate coastal chlorophyll-box placements against
the real river mouth (-65.878, 18.438, from the actual NHD flowline endpoint)
instead of the original box, which was large and poorly centered (mouth sat
in its bottom-left corner, diluting any plume signal with open-ocean noise).

Uses VIIRS only (2012-present) for a fast comparison; the winning box gets
the full MODIS+VIIRS merge treatment afterward if it shows improvement.
"""
import subprocess
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

MOUTH_LON, MOUTH_LAT = -65.878, 18.438

CANDIDATES = {
    "original":   dict(lon_min=-65.90, lon_max=-65.70, lat_min=18.35, lat_max=18.60),
    "near_tight": dict(lon_min=-65.93, lon_max=-65.85, lat_min=18.40, lat_max=18.46),
    "near_wide":  dict(lon_min=-65.95, lon_max=-65.80, lat_min=18.38, lat_max=18.50),
    "far_offshore": dict(lon_min=-65.93, lon_max=-65.83, lat_min=18.46, lat_max=18.54),
}


def fetch_viirs_box(box, label):
    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaMonthly.csv?"
        f"chlor_a[(2012-01-01):1:(2026-06-01)][(0.0)]"
        f"[({box['lat_min']}):({box['lat_max']})][({box['lon_min']}):({box['lon_max']})]"
    )
    result = subprocess.run(["curl", "-sgL", url], capture_output=True, text=True, check=True, timeout=120)
    reader = csv.reader(result.stdout.splitlines()[2:])
    by_month = defaultdict(list)
    for row in reader:
        if len(row) < 5:
            continue
        time_str, alt, lat, lon, chl = row
        if chl == "NaN":
            continue
        by_month[time_str[:7]].append(float(chl))
    n_pix = np.mean([len(v) for v in by_month.values()]) if by_month else 0
    print(f"  [{label}] {len(by_month)} months, avg {n_pix:.0f} px/month")
    return by_month


stage_a = pd.read_csv(PROC / "stage_a_monthly.csv")
stage_a["ym"] = pd.PeriodIndex(stage_a["ym"], freq="M")

print(f"River mouth (from NHD flowline): {MOUTH_LON}, {MOUTH_LAT}\n")
results = []
for label, box in CANDIDATES.items():
    print(f"Fetching box '{label}': {box}")
    by_month = fetch_viirs_box(box, label)
    if not by_month:
        continue
    chl = pd.DataFrame([(k, np.mean(v)) for k, v in by_month.items()], columns=["ym", "chl"])
    chl["ym"] = pd.PeriodIndex(chl["ym"], freq="M")
    chl["log_chl"] = np.log(chl["chl"])
    clim = chl.groupby(chl["ym"].dt.month)["log_chl"].transform("mean")
    chl["log_chl_anomaly"] = chl["log_chl"] - clim

    merged = stage_a.merge(chl[["ym", "log_chl_anomaly"]], on="ym", how="inner").dropna(
        subset=["log_q_anomaly", "log_chl_anomaly"]
    )
    best_r, best_lag = 0, 0
    for lag in range(0, 4):
        shifted = merged["log_q_anomaly"].shift(lag)
        valid = merged["log_chl_anomaly"].notna() & shifted.notna()
        r = merged.loc[valid, "log_chl_anomaly"].corr(shifted[valid])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    print(f"  -> n={len(merged)}, best |r|={abs(best_r):.3f} at lag={best_lag}\n")
    results.append({"box": label, "n_months": len(merged), "best_r": round(best_r, 3), "best_lag": best_lag})

print("=== Summary ===")
print(pd.DataFrame(results).to_string(index=False))
