"""
Pull daily precipitation for GHCN-Daily station RQC00669521
(Trujillo Alto 2 SSW, PR) — in-basin, 1957-2026 record. No auth required.
Parses the fixed-width .dly format documented at:
https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-readme.txt
"""
import urllib.request
import csv
from pathlib import Path

STATION = "RQC00669521"
URL = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{STATION}.dly"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "trujillo_alto_precip_daily.csv"

print(f"Fetching {URL}")
with urllib.request.urlopen(URL) as resp:
    lines = resp.read().decode("utf-8").splitlines()

rows = []
for line in lines:
    element = line[17:21]
    if element != "PRCP":
        continue
    year = int(line[11:15])
    month = int(line[15:17])
    for day in range(1, 32):
        offset = 21 + (day - 1) * 8
        value_str = line[offset:offset + 5]
        try:
            value = int(value_str)
        except ValueError:
            continue
        if value == -9999:
            continue  # missing
        try:
            date = f"{year:04d}-{month:02d}-{day:02d}"
            import datetime
            datetime.date(year, month, day)
        except ValueError:
            continue  # invalid day for this month (e.g. Feb 30)
        rows.append((date, value / 10.0))  # tenths of mm -> mm

rows.sort()
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "precip_mm"])
    w.writerows(rows)

print(f"Wrote {len(rows)} daily precip records to {OUT}")
if rows:
    print(f"Date range: {rows[0][0]} to {rows[-1][0]}")
