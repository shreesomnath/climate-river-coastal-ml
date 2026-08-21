"""
Phase 3, step 1: batch-fetch the mechanical inputs (discharge, river mouth)
for all 7 candidate watersheds. Precip stations handled separately (script
16) since GHCN station search needs a per-basin bounding search, not a
single API call per gauge.

Outputs one discharge CSV + one river-mouth coordinate per watershed, plus
a QC summary flagging anything that looks regulated (a sudden flow
discontinuity, near-zero flows, etc. -- the Loiza lesson: don't trust a
gauge is natural just because its name doesn't say "BLW DAM").
"""
import urllib.request
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "watersheds"
RAW.mkdir(parents=True, exist_ok=True)

# Real PR coastline (17,243-vertex OCHA boundary) -- used to find each river's
# true mouth by proximity to the coast, NOT by assuming a flow direction.
# (Caught before running: "northernmost point" only works for north-flowing
# rivers like Manati/Plata -- Fajardo flows east, Anasco west, Patillas/
# Guanajibo south. Using direction-based heuristics would have silently
# picked wrong interior points for those.)
with open(ROOT / "data" / "raw" / "pr_mainland_ocha_highres.json") as f:
    _mainland_ring = json.load(f)["mainland_ring"]
PR_COASTLINE = Polygon(_mainland_ring).exterior

CANDIDATES = {
    # river_gnis: a single distinctive keyword, not the full name -- multi-word
    # LIKE queries against this service returned 0 results (tested); single
    # keyword worked for Loiza and confirmed again here for Plata (144 hits).
    "manati":    {"gauge": "50035000", "river_gnis": "Manati",    "lat": 18.324, "lon": -66.460},
    "plata":     {"gauge": "50046000", "river_gnis": "Plata",     "lat": 18.412, "lon": -66.261},
    "fajardo":   {"gauge": "50071000", "river_gnis": "Fajardo",   "lat": 18.299, "lon": -65.694},
    "anasco":    {"gauge": "50144000", "river_gnis": "A%C3%B1asco", "lat": 18.284, "lon": -67.051},  # Anasco has a real ntilde in GNIS, %C3%B1 = URL-encoded ñ
    "patillas":  {"gauge": "50092000", "river_gnis": "Patillas",  "lat": 18.034, "lon": -66.033},
    "culebrinas":{"gauge": "50147800", "river_gnis": "Culebrinas","lat": 18.362, "lon": -67.093},
    "guanajibo": {"gauge": "50138000", "river_gnis": "Guanajibo", "lat": 18.143, "lon": -67.149},
}


def fetch_discharge(gauge_id):
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=rdb&sites={gauge_id}&parameterCd=00060&statCd=00003"
        "&startDT=1900-01-01&endDT=2026-08-20"
    )
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")
    lines = [l for l in raw.splitlines() if not l.startswith("#")]
    reader = csv.reader(lines, delimiter="\t")
    header = next(reader)
    next(reader)
    value_col = next((i for i, h in enumerate(header) if h.endswith("_00060_00003")), None)
    if value_col is None:
        return []
    rows = []
    for row in reader:
        if len(row) <= value_col or row[value_col] == "":
            continue
        rows.append((row[2], row[value_col]))
    return rows


def _query_flowlines(layer, gnis_keyword):
    url = (
        f"https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/{layer}/query?"
        f"where=GNIS_Name+LIKE+%27%25{gnis_keyword}%25%27"
        "&outFields=GNIS_Name&f=geojson"
    )
    result = subprocess.run(["curl", "-sgL", url], capture_output=True, text=True, timeout=60)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    pts = []
    for feat in data.get("features", []):
        geom = feat["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for line in lines:
            pts.extend(line)
    return pts


def fetch_river_mouth(gnis_keyword, gauge_lon, gauge_lat, max_dist_deg=0.5):
    # Try small-scale (layer 5) first, then large-scale (layer 6) -- some
    # smaller rivers (found: Fajardo, Patillas) aren't named in one or the
    # other. If NEITHER has a named match near the gauge, fall back to the
    # nearest point on the real coastline to the gauge itself, clearly
    # flagged as an approximation (documented, not silently guessed).
    for layer in (5, 6):
        pts = _query_flowlines(layer, gnis_keyword)
        pts = [p for p in pts if np.hypot(p[0] - gauge_lon, p[1] - gauge_lat) < max_dist_deg]
        if pts:
            dists = [PR_COASTLINE.distance(Point(p)) for p in pts]
            mouth = pts[int(np.argmin(dists))]
            return mouth, len(pts), f"nhd_layer{layer}"
    # Fallback: nearest point on the coastline polygon to the gauge itself
    nearest = PR_COASTLINE.interpolate(PR_COASTLINE.project(Point(gauge_lon, gauge_lat)))
    return [nearest.x, nearest.y], 0, "approximated_nearest_coast"


qc_rows = []
for name, info in CANDIDATES.items():
    print(f"\n=== {name} (gauge {info['gauge']}, {info['river_gnis']}) ===")
    rows = fetch_discharge(info["gauge"])
    if not rows:
        print("  NO DISCHARGE DATA -- skipping")
        qc_rows.append({"watershed": name, "status": "NO_DATA"})
        continue
    out = RAW / f"{name}_discharge_daily.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "discharge_cfs"])
        w.writerows(rows)
    vals = np.array([float(v) for _, v in rows if v not in ("", "Ice")])
    vals = vals[~np.isnan(vals)]
    print(f"  discharge: {len(rows)} days, {rows[0][0]} to {rows[-1][0]}, "
          f"min={vals.min():.2f} max={vals.max():.2f} mean={vals.mean():.2f} cfs")
    # crude regulation flag: many rivers have *some* near-zero days naturally in dry season,
    # but a huge fraction at exactly a repeated low constant value suggests regulated releases
    frac_near_min = np.mean(np.isclose(vals, vals.min(), atol=0.5))
    regulation_flag = frac_near_min > 0.15

    mouth, n_pts, mouth_source = fetch_river_mouth(info["river_gnis"], info["lon"], info["lat"])
    print(f"  river mouth: {mouth} ({n_pts} flowline points, source={mouth_source})")

    qc_rows.append({
        "watershed": name, "gauge": info["gauge"], "status": "OK",
        "n_days": len(rows), "start": rows[0][0], "end": rows[-1][0],
        "min_cfs": round(vals.min(), 2), "max_cfs": round(vals.max(), 2), "mean_cfs": round(vals.mean(), 2),
        "frac_near_min": round(frac_near_min, 3), "possible_regulation": regulation_flag,
        "mouth_lon": mouth[0] if mouth else None, "mouth_lat": mouth[1] if mouth else None,
        "n_flowline_pts": n_pts, "mouth_source": mouth_source,
    })

qc = pd.DataFrame(qc_rows)
qc.to_csv(RAW / "watershed_qc_summary.csv", index=False)
print("\n\n=== QC SUMMARY ===")
print(qc.to_string(index=False))
