"""
Scaling the basin set further (n=8 -> n=11): Humacao, Guayanes, Coamo.
Same full pipeline as before (discharge, regulation QC, river mouth via
NHD/coastline-distance, precip station, SPI, Stage A model) -- reuses the
exact methods already validated, just applied to 3 more basins.
"""
import urllib.request
import csv
import json
import subprocess
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import Point, Polygon
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW_W = RAW / "watersheds"
PROC_W = ROOT / "data" / "processed" / "watersheds"

with open(RAW / "pr_mainland_ocha_highres.json") as f:
    PR_COASTLINE = Polygon(json.load(f)["mainland_ring"]).exterior

NEW_BASINS = {
    "humacao":  {"gauge": "50081000", "river_gnis": "Humacao",  "lat": 18.174, "lon": -65.869},
    "guayanes": {"gauge": "50083500", "river_gnis": "Guayanes", "lat": 18.059, "lon": -65.901},
    "coamo":    {"gauge": "50106100", "river_gnis": "Coamo",    "lat": 18.084, "lon": -66.355},
}


def fetch_discharge(gauge_id):
    url = (f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={gauge_id}"
           "&parameterCd=00060&statCd=00003&startDT=1900-01-01&endDT=2026-08-20")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")
    lines = [l for l in raw.splitlines() if not l.startswith("#")]
    reader = csv.reader(lines, delimiter="\t")
    header = next(reader); next(reader)
    vcol = next((i for i, h in enumerate(header) if h.endswith("_00060_00003")), None)
    if vcol is None:
        return []
    return [(row[2], row[vcol]) for row in reader if len(row) > vcol and row[vcol] != ""]


def _query_flowlines(layer, kw):
    url = (f"https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/{layer}/query?"
           f"where=GNIS_Name+LIKE+%27%25{kw}%25%27&outFields=GNIS_Name&f=geojson")
    r = subprocess.run(["curl", "-sgL", url], capture_output=True, text=True, timeout=60)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    pts = []
    for feat in data.get("features", []):
        geom = feat["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for line in lines:
            pts.extend(line)
    return pts


def fetch_mouth(kw, glon, glat, max_dist=0.5):
    for layer in (5, 6):
        pts = [p for p in _query_flowlines(layer, kw) if np.hypot(p[0]-glon, p[1]-glat) < max_dist]
        if pts:
            dists = [PR_COASTLINE.distance(Point(p)) for p in pts]
            return pts[int(np.argmin(dists))], f"nhd_layer{layer}"
    nearest = PR_COASTLINE.interpolate(PR_COASTLINE.project(Point(glon, glat)))
    return [nearest.x, nearest.y], "approximated"


def fetch_precip_dly(station_id):
    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
    with urllib.request.urlopen(url) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    rows = []
    for line in lines:
        if line[17:21] != "PRCP":
            continue
        year, month = int(line[11:15]), int(line[15:17])
        for day in range(1, 32):
            off = 21 + (day-1)*8
            try:
                v = int(line[off:off+5])
            except ValueError:
                continue
            if v == -9999:
                continue
            try:
                datetime.date(year, month, day)
            except ValueError:
                continue
            rows.append((f"{year:04d}-{month:02d}-{day:02d}", v/10.0))
    return pd.DataFrame(rows, columns=["date", "precip_mm"])


def spi(series, window):
    accum = series.rolling(window, min_periods=window).sum()
    out = pd.Series(index=series.index, dtype=float)
    for month in range(1, 13):
        vals = accum[accum.index.month == month].dropna()
        if len(vals) < 10:
            continue
        p_zero = (vals == 0).sum() / len(vals)
        nz = vals[vals > 0]
        if len(nz) < 5:
            continue
        sh, lo, sc = stats.gamma.fit(nz, floc=0)
        cdf = np.where(vals == 0, p_zero, p_zero + (1-p_zero)*stats.gamma.cdf(vals, sh, loc=lo, scale=sc))
        out.loc[vals.index] = stats.norm.ppf(np.clip(cdf, 1e-6, 1-1e-6))
    return out


def nse(o, p):
    return 1 - np.sum((o-p)**2) / np.sum((o-o.mean())**2)


# ---- station inventory (reuse cached files) ----
stations = []
with open("/tmp/ghcnd-stations.txt") as f:
    for line in f:
        sid = line[0:11].strip()
        if sid.startswith("RQ"):
            stations.append((sid, float(line[12:20]), float(line[21:30]), line[41:71].strip()))
stations_df = pd.DataFrame(stations, columns=["id", "lat", "lon", "name"])
inv = {}
with open("/tmp/ghcnd-inventory.txt") as f:
    for line in f:
        parts = line.split()
        if len(parts) < 6 or parts[3] != "PRCP":
            continue
        inv[parts[0]] = (int(parts[4]), int(parts[5]))
stations_df["prcp_start"] = stations_df["id"].map(lambda s: inv.get(s, (None, None))[0])
stations_df["prcp_end"] = stations_df["id"].map(lambda s: inv.get(s, (None, None))[1])
stations_df = stations_df.dropna(subset=["prcp_start"])
stations_df["record_len"] = stations_df["prcp_end"] - stations_df["prcp_start"]

FEATURES = ["spi_1", "spi_3", "spi_6", "spi_12", "precip_mm_sum", "month_sin", "month_cos", "log_q_anomaly_lag1"]
TARGET = "log_q_anomaly"
results = []

for name, info in NEW_BASINS.items():
    print(f"\n=== {name} ===")
    rows = fetch_discharge(info["gauge"])
    out = RAW_W / f"{name}_discharge_daily.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "discharge_cfs"]); w.writerows(rows)
    vals = np.array([float(v) for _, v in rows if v not in ("", "Ice")])
    vals = vals[~np.isnan(vals)]
    print(f"  discharge: {len(rows)} days, {rows[0][0]} to {rows[-1][0]}, mean={vals.mean():.1f} cfs")

    mouth, src = fetch_mouth(info["river_gnis"], info["lon"], info["lat"])
    print(f"  mouth: {mouth} ({src})")

    stations_df["dist_deg"] = np.hypot(stations_df["lat"]-info["lat"], stations_df["lon"]-info["lon"])
    nearby = stations_df[(stations_df["dist_deg"] < 0.15) & (stations_df["prcp_end"] >= 2020)].sort_values("record_len", ascending=False)
    if nearby.empty:
        print("  NO PRECIP STATION -- skipping")
        continue
    st = nearby.iloc[0]
    print(f"  precip station: {st['id']} {st['name']} ({st['prcp_start']:.0f}-{st['prcp_end']:.0f})")
    p = fetch_precip_dly(st["id"])
    p["date"] = pd.to_datetime(p["date"])
    p.to_csv(RAW_W / f"{name}_precip_daily.csv", index=False)

    q = pd.DataFrame(rows, columns=["date", "discharge_cfs"])
    q["date"] = pd.to_datetime(q["date"])
    q = q[pd.to_numeric(q["discharge_cfs"], errors="coerce") > 0].copy()
    q["discharge_cfs"] = pd.to_numeric(q["discharge_cfs"])
    q["ym"] = q["date"].dt.to_period("M")
    q_monthly = q.groupby("ym")["discharge_cfs"].mean().to_frame("discharge_cfs_mean")
    q_full = pd.period_range(q_monthly.index.min(), q_monthly.index.max(), freq="M")
    q_monthly = q_monthly.reindex(q_full)
    q_monthly["log_q"] = np.log(q_monthly["discharge_cfs_mean"])
    clim_q = q_monthly.groupby(q_monthly.index.month)["log_q"].transform("mean")
    q_monthly["log_q_anomaly"] = q_monthly["log_q"] - clim_q

    p["ym"] = p["date"].dt.to_period("M")
    p_monthly = p.groupby("ym")["precip_mm"].sum().to_frame("precip_mm_sum")
    coverage = p.groupby("ym")["precip_mm"].count()
    p_monthly.loc[coverage[coverage < 20].index, "precip_mm_sum"] = np.nan
    p_full = pd.period_range(p_monthly.index.min(), p_monthly.index.max(), freq="M")
    p_monthly = p_monthly.reindex(p_full)
    for w in (1, 3, 6, 12):
        p_monthly[f"spi_{w}"] = spi(p_monthly["precip_mm_sum"], w)
    p_monthly["month_sin"] = np.sin(2*np.pi*p_monthly.index.month/12)
    p_monthly["month_cos"] = np.cos(2*np.pi*p_monthly.index.month/12)

    df = q_monthly.join(p_monthly, how="inner")
    df["log_q_anomaly_lag1"] = df["log_q_anomaly"].shift(1)
    df.to_csv(PROC_W / f"{name}_stage_a_monthly.csv")

    df_valid = df.dropna(subset=FEATURES + [TARGET])
    if len(df_valid) < 60:
        print(f"  INSUFFICIENT ({len(df_valid)} rows)")
        continue
    split = int(len(df_valid) * 0.8)
    train, test = df_valid.iloc[:split], df_valid.iloc[split:]
    scaler = StandardScaler().fit(train[FEATURES])
    en = ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), random_state=0).fit(scaler.transform(train[FEATURES]), train[TARGET])
    en_pred = en.predict(scaler.transform(test[FEATURES]))
    rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=4, random_state=0)
    rf.fit(train[FEATURES], train[TARGET])
    rf_pred = rf.predict(test[FEATURES])
    nse_clim = nse(test[TARGET].values, np.zeros(len(test)))
    nse_en = nse(test[TARGET].values, en_pred)
    print(f"  n={len(df_valid)} ({df_valid.index.min()} to {df_valid.index.max()}), "
          f"NSE_clim={nse_clim:+.3f} NSE_EN={nse_en:+.3f} NSE_RF={nse(test[TARGET].values, rf_pred):+.3f}")
    results.append({"watershed": name, "mouth_lon": mouth[0], "mouth_lat": mouth[1], "mouth_source": src,
                     "n": len(df_valid), "NSE_climatology": round(nse_clim, 3), "NSE_ElasticNet": round(nse_en, 3),
                     "NSE_RandomForest": round(nse(test[TARGET].values, rf_pred), 3)})

pd.DataFrame(results).to_csv(PROC_W / "stage_a_3more_watersheds.csv", index=False)
print("\n\n=== NEW BASINS SUMMARY ===")
print(pd.DataFrame(results).to_string(index=False))
