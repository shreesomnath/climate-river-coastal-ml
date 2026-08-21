"""
Phase 3, step 2: find the best long-record GHCN precipitation station near
each candidate watershed's gauge, same selection logic used for Loiza
(prefer in-basin proximity + record length overlapping the discharge
record, over just picking the single longest-record station on the island).
"""
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "watersheds"

GAUGES = {
    "manati":    (18.324, -66.460),
    "plata":     (18.412, -66.261),
    "fajardo":   (18.299, -65.694),
    "anasco":    (18.284, -67.051),
    "patillas":  (18.034, -66.033),
    "culebrinas":(18.362, -67.093),
    "guanajibo": (18.143, -67.149),
}

# ---- Load station inventory once ----
stations = []
with open("/tmp/ghcnd-stations.txt") as f:
    for line in f:
        sid = line[0:11].strip()
        lat = float(line[12:20])
        lon = float(line[21:30])
        name = line[41:71].strip()
        if sid.startswith("RQ"):  # Puerto Rico stations only
            stations.append((sid, lat, lon, name))
stations_df = pd.DataFrame(stations, columns=["id", "lat", "lon", "name"])

inventory = {}
with open("/tmp/ghcnd-inventory.txt") as f:
    for line in f:
        parts = line.split()
        if len(parts) < 6 or parts[3] != "PRCP":
            continue
        sid = parts[0]
        first_year, last_year = int(parts[4]), int(parts[5])
        inventory[sid] = (first_year, last_year)

stations_df["prcp_start"] = stations_df["id"].map(lambda s: inventory.get(s, (None, None))[0])
stations_df["prcp_end"] = stations_df["id"].map(lambda s: inventory.get(s, (None, None))[1])
stations_df = stations_df.dropna(subset=["prcp_start"])
stations_df["record_len"] = stations_df["prcp_end"] - stations_df["prcp_start"]

results = []
for name, (glat, glon) in GAUGES.items():
    stations_df["dist_deg"] = np.hypot(stations_df["lat"] - glat, stations_df["lon"] - glon)
    nearby = stations_df[stations_df["dist_deg"] < 0.15].copy()  # ~16km radius
    # long record AND recent (still reporting toward present) AND close
    nearby = nearby[nearby["prcp_end"] >= 2020]
    nearby = nearby.sort_values("record_len", ascending=False)
    if nearby.empty:
        print(f"{name}: NO SUITABLE STATION within 0.15 deg with data through 2020+")
        results.append({"watershed": name, "station_id": None})
        continue
    best = nearby.iloc[0]
    print(f"{name}: {best['id']} {best['name']} | {best['prcp_start']:.0f}-{best['prcp_end']:.0f} "
          f"({best['record_len']:.0f} yr) | {best['dist_deg']*111:.1f} km from gauge")
    results.append({
        "watershed": name, "station_id": best["id"], "station_name": best["name"],
        "lat": best["lat"], "lon": best["lon"],
        "prcp_start": int(best["prcp_start"]), "prcp_end": int(best["prcp_end"]),
        "dist_km": round(best["dist_deg"] * 111, 1),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(RAW / "precip_stations_selected.csv", index=False)
print(f"\nSaved to {RAW / 'precip_stations_selected.csv'}")
