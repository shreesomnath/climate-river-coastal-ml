"""
Stage B, step 1 for the 7 new watersheds: quick VIIRS-only correlation
screen, near-mouth vs. offshore box, per basin -- same logic as
11_test_coastal_boxes.py for Loiza, generalized. Do NOT assume Loiza's
"offshore beats near-mouth" finding generalizes; each basin gets its own
test.

"Offshore" direction is computed per-basin as the radial direction from the
PR mainland polygon's centroid through the river mouth (robust across N/S/
E/W coasts without hand-reasoning compass directions per river).
"""
import json
import subprocess
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW_W = RAW / "watersheds"
PROC = ROOT / "data" / "processed"
PROC_W = PROC / "watersheds"

qc = pd.read_csv(RAW_W / "watershed_qc_summary.csv")
with open(RAW / "pr_mainland_ocha_highres.json") as f:
    mainland = Polygon(json.load(f)["mainland_ring"])
centroid = mainland.centroid

BOX_HALF_SIZE = 0.05  # ~5.5km half-width


def make_box(lon, lat, half=BOX_HALF_SIZE):
    return dict(lon_min=lon - half, lon_max=lon + half, lat_min=lat - half, lat_max=lat + half)


def fetch_viirs_box(box):
    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaMonthly.csv?"
        f"chlor_a[(2012-01-01):1:(2026-06-01)][(0.0)]"
        f"[({box['lat_min']}):({box['lat_max']})][({box['lon_min']}):({box['lon_max']})]"
    )
    result = subprocess.run(["curl", "-sgL", url], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return {}
    reader = csv.reader(result.stdout.splitlines()[2:])
    by_month = defaultdict(list)
    for row in reader:
        if len(row) < 5 or row[-1] == "NaN":
            continue
        by_month[row[0][:7]].append(float(row[-1]))
    return by_month


results = []
for _, row in qc.iterrows():
    name = row["watershed"]
    if pd.isna(row["mouth_lon"]):
        continue
    mouth_lon, mouth_lat = row["mouth_lon"], row["mouth_lat"]

    # radial offshore direction: centroid -> mouth, extended further out
    dx, dy = mouth_lon - centroid.x, mouth_lat - centroid.y
    norm = np.hypot(dx, dy)
    offshore_lon = mouth_lon + (dx / norm) * 0.09
    offshore_lat = mouth_lat + (dy / norm) * 0.09

    stage_a_path = PROC_W / f"{name}_stage_a_monthly.csv"
    stage_a = pd.read_csv(stage_a_path)
    stage_a["ym"] = pd.PeriodIndex(stage_a[stage_a.columns[0]], freq="M")

    print(f"\n=== {name} (mouth: {mouth_lon:.3f},{mouth_lat:.3f} | offshore test point: "
          f"{offshore_lon:.3f},{offshore_lat:.3f}) ===")

    for label, (blon, blat) in [("near_mouth", (mouth_lon, mouth_lat)), ("offshore", (offshore_lon, offshore_lat))]:
        box = make_box(blon, blat)
        by_month = fetch_viirs_box(box)
        if not by_month:
            print(f"  [{label}] no data returned")
            results.append({"watershed": name, "box": label, "n_months": 0, "best_r": None})
            continue
        chl = pd.DataFrame([(k, np.mean(v)) for k, v in by_month.items()], columns=["ym", "chl"])
        chl["ym"] = pd.PeriodIndex(chl["ym"], freq="M")
        chl["log_chl"] = np.log(chl["chl"])
        clim = chl.groupby(chl["ym"].dt.month)["log_chl"].transform("mean")
        chl["log_chl_anomaly"] = chl["log_chl"] - clim

        merged = stage_a.merge(chl[["ym", "log_chl_anomaly"]], on="ym", how="inner").dropna(
            subset=["log_q_anomaly", "log_chl_anomaly"])
        if len(merged) < 24:
            print(f"  [{label}] only {len(merged)} overlapping months -- too few")
            results.append({"watershed": name, "box": label, "n_months": len(merged), "best_r": None})
            continue
        best_r, best_lag = 0, 0
        for lag in range(0, 4):
            shifted = merged["log_q_anomaly"].shift(lag)
            valid = merged["log_chl_anomaly"].notna() & shifted.notna()
            if valid.sum() < 10:
                continue
            r = merged.loc[valid, "log_chl_anomaly"].corr(shifted[valid])
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag
        print(f"  [{label}] n={len(merged)}, best |r|={abs(best_r):.3f} at lag={best_lag}")
        results.append({"watershed": name, "box": label, "n_months": len(merged),
                         "best_r": round(best_r, 3), "best_lag": best_lag})

results_df = pd.DataFrame(results)
results_df.to_csv(PROC_W / "coastal_box_screen_all_watersheds.csv", index=False)
print("\n\n=== SUMMARY ===")
print(results_df.to_string(index=False))
