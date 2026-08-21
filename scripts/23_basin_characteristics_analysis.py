"""
"Why does the discharge->chlorophyll signal appear in some basins and not
others?" -- compare basin physical characteristics (drainage area, mean
discharge, coastal shelf steepness) against the Stage B result.

Honest caveat up front: n=8 basins is too small for real statistical
inference (correlations here are descriptive/exploratory, not hypothesis
tests) -- but the pattern is still worth looking at as a lead for future
work / a paper discussion point, which is what this analysis is for.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW_W = RAW / "watersheds"
PROC = ROOT / "data" / "processed"
PROC_W = PROC / "watersheds"

# ---- Drainage area (USGS NWIS, real data) ----
DRAIN_AREA_MI2 = {
    "loiza": 89.8, "manati": 134.0, "plata": 208.0, "fajardo": 14.9,
    "anasco": 134.0, "patillas": 18.3, "culebrinas": 71.2, "guanajibo": 120.0,
}

# ---- Mean discharge (from raw daily discharge files) ----
mean_discharge = {}
loiza_q = pd.read_csv(RAW / "loiza_caguas_discharge_daily.csv")
mean_discharge["loiza"] = pd.to_numeric(loiza_q["discharge_cfs"], errors="coerce").mean()
for name in ["manati", "plata", "fajardo", "anasco", "patillas", "culebrinas", "guanajibo"]:
    q = pd.read_csv(RAW_W / f"{name}_discharge_daily.csv")
    mean_discharge[name] = pd.to_numeric(q["discharge_cfs"], errors="coerce").mean()

# ---- River mouths ----
qc = pd.read_csv(RAW_W / "watershed_qc_summary.csv").set_index("watershed")
MOUTHS = {row: (qc.loc[row, "mouth_lon"], qc.loc[row, "mouth_lat"]) for row in qc.index if pd.notna(qc.loc[row, "mouth_lon"])}
MOUTHS["loiza"] = (-65.878, 18.438)
# Fajardo's corrected OSM mouth (QC file has this updated already per earlier fix)

# ---- Shelf steepness proxy: distance from mouth to the -20m isobath ----
elev = pd.read_csv(RAW / "gebco_2020_pr.csv", skiprows=[1])
elev_grid = elev.pivot(index="latitude", columns="longitude", values="elevation")
lats, lons = elev_grid.index.values, elev_grid.columns.values
Z = elev_grid.values


def dist_to_isobath(mouth_lon, mouth_lat, target_depth=-20, search_radius_deg=0.3):
    lon_mask = (lons > mouth_lon - search_radius_deg) & (lons < mouth_lon + search_radius_deg)
    lat_mask = (lats > mouth_lat - search_radius_deg) & (lats < mouth_lat + search_radius_deg)
    sub_lons, sub_lats = lons[lon_mask], lats[lat_mask]
    sub_Z = Z[np.ix_(lat_mask, lon_mask)]
    # candidate cells near the target depth
    close = np.abs(sub_Z - target_depth) < 5
    if not close.any():
        return np.nan
    lat_idx, lon_idx = np.where(close)
    cand_lats, cand_lons = sub_lats[lat_idx], sub_lons[lon_idx]
    dists_km = np.hypot((cand_lons - mouth_lon) * 111 * np.cos(np.radians(mouth_lat)),
                         (cand_lats - mouth_lat) * 111)
    return dists_km.min()


shelf_dist = {name: dist_to_isobath(lon, lat) for name, (lon, lat) in MOUTHS.items()}

# ---- Stage B result (final holdout NSE, ElasticNet; from prior run) ----
STAGE_B_NSE = {
    "loiza": 0.029, "manati": 0.146, "patillas": 0.096, "anasco": 0.063,
    "plata": 0.030, "culebrinas": -0.059, "guanajibo": -0.174, "fajardo": np.nan,
}
VERDICT = {
    "loiza": "Promising", "manati": "Promising", "patillas": "Weak/mixed", "anasco": "Weak/mixed",
    "plata": "Null", "culebrinas": "Null", "guanajibo": "Null", "fajardo": "Untestable",
}

rows = []
for name in DRAIN_AREA_MI2:
    rows.append({
        "basin": name, "drainage_area_mi2": DRAIN_AREA_MI2[name],
        "mean_discharge_cfs": round(mean_discharge.get(name, np.nan), 1),
        "shelf_dist_km_to_neg20m": round(shelf_dist.get(name, np.nan), 2),
        "stage_b_NSE": STAGE_B_NSE[name], "verdict": VERDICT[name],
    })
df = pd.DataFrame(rows).sort_values("stage_b_NSE", ascending=False)
print(df.to_string(index=False))
df.to_csv(PROC_W / "basin_characteristics_vs_stage_b.csv", index=False)

# ---- Descriptive correlation (n=7, excludes Fajardo -- untestable) ----
testable = df.dropna(subset=["stage_b_NSE"])
print(f"\nn={len(testable)} testable basins")
for col in ["drainage_area_mi2", "mean_discharge_cfs", "shelf_dist_km_to_neg20m"]:
    r = testable[col].corr(testable["stage_b_NSE"])
    print(f"  corr(stage_b_NSE, {col}) = {r:+.3f}")
