"""
Pull full daily discharge history for Rio Grande de Loiza at Caguas, PR
(USGS gauge 50055000) from the public NWIS web service. No auth required.
"""
import urllib.request
import csv
import io
from pathlib import Path

SITE = "50055000"
PARAM = "00060"  # discharge, cubic feet per second
STAT = "00003"   # daily mean
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "loiza_caguas_discharge_daily.csv"

url = (
    "https://waterservices.usgs.gov/nwis/dv/"
    f"?format=rdb&sites={SITE}&parameterCd={PARAM}&statCd={STAT}"
    "&startDT=1959-01-01&endDT=2026-08-20"
)

print(f"Fetching {url}")
with urllib.request.urlopen(url) as resp:
    raw = resp.read().decode("utf-8")

lines = [l for l in raw.splitlines() if not l.startswith("#")]
reader = csv.reader(lines, delimiter="\t")
header = next(reader)
next(reader)  # rdb format spec line (5s 15s 20d ...)

value_col = next(i for i, h in enumerate(header) if h.endswith("_00060_00003"))
cd_col = value_col + 1

rows = []
for row in reader:
    if len(row) <= value_col:
        continue
    date, value = row[2], row[value_col]
    qual = row[cd_col] if len(row) > cd_col else ""
    if value == "":
        continue
    rows.append((date, value, qual))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "discharge_cfs", "qual_code"])
    w.writerows(rows)

print(f"Wrote {len(rows)} daily records to {OUT}")
print(f"Date range: {rows[0][0]} to {rows[-1][0]}")
