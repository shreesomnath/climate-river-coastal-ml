"""
Pull MODIS-Aqua monthly chlorophyll-a (NOAA CoastWatch ERDDAP, dataset
erdMH1chlamday, public, no auth) for a coastal box near the Rio Grande de
Loiza river mouth (Pinones / San Juan Bay area, north coast of PR).

Dataset record: 2003-01-16 to 2022-03-16 (~19 yr) -- this is the limiting
factor for Stage B's usable period (see modeling_spec.md). Land pixels come
back as NaN and are dropped when averaging each monthly slice.
"""
import urllib.request
import csv
from pathlib import Path
from collections import defaultdict

LAT_MIN, LAT_MAX = 18.35, 18.60
LON_MIN, LON_MAX = -65.90, -65.70

URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday.csv?"
    f"chlorophyll[(2003-01-01):1:(2022-03-16)]"
    f"[({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"
)
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "loiza_mouth_chlorophyll_monthly.csv"

print(f"Fetching {URL}")
with urllib.request.urlopen(URL) as resp:
    lines = resp.read().decode("utf-8").splitlines()

reader = csv.reader(lines[2:])  # skip header row + units row
by_month = defaultdict(list)
n_total, n_nan = 0, 0
for row in reader:
    if len(row) < 4:
        continue
    time_str, lat, lon, chl = row
    n_total += 1
    if chl == "NaN":
        n_nan += 1
        continue
    month = time_str[:7]  # YYYY-MM
    by_month[month].append(float(chl))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["year_month", "chl_a_mg_m3_mean", "n_valid_pixels"])
    for month in sorted(by_month):
        vals = by_month[month]
        w.writerow([month, sum(vals) / len(vals), len(vals)])

print(f"Pixels: {n_total} total, {n_nan} NaN (land-masked), {n_total - n_nan} ocean")
print(f"Wrote {len(by_month)} monthly chlorophyll-a means to {OUT}")
