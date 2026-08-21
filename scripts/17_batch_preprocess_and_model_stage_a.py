"""
Phase 3, step 3: SPI + Stage A (climate -> discharge) for all 7 candidate
watersheds, generalizing scripts 03+04. Each basin uses its OWN full
available record as the primary analysis (standard practice, maximizes
power -- all 7 records extend to 2026 regardless of start year, so they
already share the same recent decades/events; they only differ in how far
back they go). A common-window (1973-2026, the latest start date among the
7 = Guanajibo) sensitivity re-run is done separately afterward specifically
to check results aren't an artifact of using different historical eras.
"""
import urllib.request
import csv
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "watersheds"
PROC = ROOT / "data" / "processed" / "watersheds"
PROC.mkdir(parents=True, exist_ok=True)

STATIONS = {
    "manati": "RQC00665807", "plata": "RQC00669415", "fajardo": "RQC00663657",
    "anasco": "RQC00662801", "patillas": "RQC00664193", "culebrinas": "RQC00662801",
    "guanajibo": "RQC00665097",
}


def fetch_precip(station_id):
    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"
    with urllib.request.urlopen(url) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    rows = []
    for line in lines:
        if line[17:21] != "PRCP":
            continue
        year, month = int(line[11:15]), int(line[15:17])
        for day in range(1, 32):
            offset = 21 + (day - 1) * 8
            try:
                value = int(line[offset:offset + 5])
            except ValueError:
                continue
            if value == -9999:
                continue
            try:
                datetime.date(year, month, day)
            except ValueError:
                continue
            rows.append((f"{year:04d}-{month:02d}-{day:02d}", value / 10.0))
    return pd.DataFrame(rows, columns=["date", "precip_mm"])


def spi(series, window):
    accum = series.rolling(window, min_periods=window).sum()
    out = pd.Series(index=series.index, dtype=float)
    for month in range(1, 13):
        mask = accum.index.month == month
        vals = accum[mask].dropna()
        if len(vals) < 10:
            continue
        zeros = (vals == 0).sum()
        p_zero = zeros / len(vals)
        nonzero = vals[vals > 0]
        if len(nonzero) < 5:
            continue
        shape_, loc_, scale_ = stats.gamma.fit(nonzero, floc=0)
        cdf = np.where(vals == 0, p_zero, p_zero + (1 - p_zero) * stats.gamma.cdf(vals, shape_, loc=loc_, scale=scale_))
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        out.loc[vals.index] = stats.norm.ppf(cdf)
    return out


def nse(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)


results = []
for name, station_id in STATIONS.items():
    print(f"\n=== {name} ===")
    # Discharge
    q = pd.read_csv(RAW / f"{name}_discharge_daily.csv", parse_dates=["date"])
    q = q[pd.to_numeric(q["discharge_cfs"], errors="coerce") > 0].copy()
    q["discharge_cfs"] = pd.to_numeric(q["discharge_cfs"])
    q["ym"] = q["date"].dt.to_period("M")
    q_monthly = q.groupby("ym")["discharge_cfs"].mean().to_frame("discharge_cfs_mean")
    # Same continuous-calendar fix as precip -- shift(1) below must mean "1
    # calendar month back," not "1 row back," which requires no absent rows.
    q_full_months = pd.period_range(q_monthly.index.min(), q_monthly.index.max(), freq="M")
    n_q_gaps = len(q_full_months.difference(q_monthly.index))
    if n_q_gaps:
        print(f"  discharge record has {n_q_gaps} calendar-gap month(s) -- reindexing to expose them as NaN")
    q_monthly = q_monthly.reindex(q_full_months)
    q_monthly["log_q"] = np.log(q_monthly["discharge_cfs_mean"])
    clim_q = q_monthly.groupby(q_monthly.index.month)["log_q"].transform("mean")
    q_monthly["log_q_anomaly"] = q_monthly["log_q"] - clim_q

    # Precip -> SPI (fetch fresh if not already saved)
    precip_path = RAW / f"{name}_precip_daily.csv"
    if precip_path.exists():
        p = pd.read_csv(precip_path, parse_dates=["date"])
    else:
        p = fetch_precip(station_id)
        p["date"] = pd.to_datetime(p["date"])
        p.to_csv(precip_path, index=False)
    p["ym"] = p["date"].dt.to_period("M")
    p_monthly = p.groupby("ym")["precip_mm"].sum().to_frame("precip_mm_sum")
    coverage = p.groupby("ym")["precip_mm"].count()
    p_monthly.loc[coverage[coverage < 20].index, "precip_mm_sum"] = np.nan
    p_monthly = p_monthly.sort_index()
    # CORRECTNESS FIX: reindex to a continuous monthly calendar before any
    # rolling-window computation. Without this, a calendar gap (a month with
    # zero raw daily reports) is simply an ABSENT row, not a NaN row --
    # pandas .rolling() operates on row position, not calendar time, so a
    # gap would silently let the window splice non-adjacent months together
    # as if consecutive instead of correctly breaking / propagating NaN
    # across the gap. Found via Fajardo's real ~5-year station gap
    # (1996-02 onward) exposing this; applies to every basin, not just
    # Fajardo -- gaps just happened to be small/rare enough elsewhere to be
    # easy to miss without this fix.
    full_months = pd.period_range(p_monthly.index.min(), p_monthly.index.max(), freq="M")
    n_gap_months = len(full_months.difference(p_monthly.index))
    if n_gap_months:
        print(f"  precip record has {n_gap_months} calendar-gap month(s) -- reindexing to expose them as NaN")
    p_monthly = p_monthly.reindex(full_months)
    for w in (1, 3, 6, 12):
        p_monthly[f"spi_{w}"] = spi(p_monthly["precip_mm_sum"], w)
    p_monthly["month_sin"] = np.sin(2 * np.pi * p_monthly.index.month / 12)
    p_monthly["month_cos"] = np.cos(2 * np.pi * p_monthly.index.month / 12)

    df = q_monthly.join(p_monthly, how="inner")
    df["log_q_anomaly_lag1"] = df["log_q_anomaly"].shift(1)
    df.to_csv(PROC / f"{name}_stage_a_monthly.csv")

    FEATURES = ["spi_1", "spi_3", "spi_6", "spi_12", "precip_mm_sum", "month_sin", "month_cos", "log_q_anomaly_lag1"]
    TARGET = "log_q_anomaly"
    df_valid = df.dropna(subset=FEATURES + [TARGET])
    if len(df_valid) < 60:
        print(f"  SKIPPING -- only {len(df_valid)} usable rows")
        results.append({"watershed": name, "status": "INSUFFICIENT_DATA", "n": len(df_valid)})
        continue

    split = int(len(df_valid) * 0.8)
    train, test = df_valid.iloc[:split], df_valid.iloc[split:]
    Xtr, ytr = train[FEATURES].values, train[TARGET].values
    Xte, yte = test[FEATURES].values, test[TARGET].values

    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), random_state=0).fit(scaler.transform(Xtr), ytr)
    en_pred = en.predict(scaler.transform(Xte))
    rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=4, random_state=0)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)

    nse_clim = nse(yte, np.zeros_like(yte))
    nse_persist = nse(yte, test["log_q_anomaly_lag1"].values)
    nse_en = nse(yte, en_pred)
    nse_rf = nse(yte, rf_pred)

    print(f"  n={len(df_valid)} ({df_valid.index.min()} to {df_valid.index.max()}), test n={len(test)}")
    print(f"  NSE: climatology={nse_clim:+.3f} persistence={nse_persist:+.3f} "
          f"ElasticNet={nse_en:+.3f} RandomForest={nse_rf:+.3f}")

    results.append({
        "watershed": name, "status": "OK", "n_total": len(df_valid),
        "record_start": str(df_valid.index.min()), "record_end": str(df_valid.index.max()),
        "n_test": len(test), "NSE_climatology": round(nse_clim, 3), "NSE_persistence": round(nse_persist, 3),
        "NSE_ElasticNet": round(nse_en, 3), "NSE_RandomForest": round(nse_rf, 3),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(PROC / "stage_a_all_watersheds_comparison.csv", index=False)
print("\n\n=== STAGE A: ALL WATERSHEDS COMPARISON ===")
print(results_df.to_string(index=False))
