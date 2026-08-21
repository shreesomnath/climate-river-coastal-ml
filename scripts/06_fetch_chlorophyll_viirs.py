"""
Pull VIIRS S-NPP monthly chlorophyll-a (NOAA CoastWatch ERDDAP, dataset
nesdisVHNSQchlaMonthly, public, no auth), 2012-present, for the same
coastal box near the Rio Grande de Loiza river mouth used in script 05.

This extends coverage past the legacy MODIS product's 2022-03 cutoff.
Variable name here is `chlor_a` (not `chlorophyll`) and the dataset has an
extra altitude dimension MODIS's erdMH1chlamday does not.

NOTE: MODIS (erdMH1chlamday) and VIIRS (this script) are different sensors
with different retrieval algorithms. Do not naively concatenate the two
raw series -- see docs/modeling_spec.md note on OC-CCI / sensor merging.
Use this file either on its own (2012-present, shorter but current) or as
input to a proper overlap-period bias correction against the MODIS series
before splicing.
"""
import subprocess
import csv
from pathlib import Path
from collections import defaultdict

LAT_MIN, LAT_MAX = 18.35, 18.60
LON_MIN, LON_MAX = -65.90, -65.70

URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaMonthly.csv?"
    f"chlor_a[(2012-01-01):1:(2026-06-01)][(0.0)]"
    f"[({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"
)
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "loiza_mouth_chlorophyll_viirs_monthly.csv"

print(f"Fetching {URL}")
result = subprocess.run(
    ["curl", "-sgL", URL], capture_output=True, text=True, check=True, timeout=120
)
lines = result.stdout.splitlines()

reader = csv.reader(lines[2:])
by_month = defaultdict(list)
n_total, n_nan = 0, 0
for row in reader:
    if len(row) < 5:
        continue
    time_str, alt, lat, lon, chl = row
    n_total += 1
    if chl == "NaN":
        n_nan += 1
        continue
    month = time_str[:7]
    by_month[month].append(float(chl))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["year_month", "chl_a_mg_m3_mean", "n_valid_pixels"])
    for month in sorted(by_month):
        vals = by_month[month]
        w.writerow([month, sum(vals) / len(vals), len(vals)])

print(f"Pixels: {n_total} total, {n_nan} NaN, {n_total - n_nan} ocean")
print(f"Wrote {len(by_month)} monthly VIIRS chlorophyll-a means to {OUT}")
