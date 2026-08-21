"""
Extend the SST/wind covariate treatment (proven to help Loiza) to Manati,
Patillas, and Anasco -- the 'promising'/'weak-mixed' basins where an extra
predictor might tip a marginal result, using each basin's own Stage B box
(saved in *_stage_b_monthly.csv from script 22).
"""
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC_W = ROOT / "data" / "processed" / "watersheds"

BASINS = {
    "manati":   dict(box_center=(-66.551, 18.569), half=0.05),
    "patillas": dict(box_center=(-65.934, 17.939), half=0.05),
    "anasco":   dict(box_center=(-67.278, 18.272), half=0.05),
}
FEATURES_BASE = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
                  "spi_3", "month_sin", "month_cos"]
TARGET = "log_chl_anomaly"

u = xr.open_dataset(RAW / "uwnd_10m_mon.nc")
v = xr.open_dataset(RAW / "vwnd_10m_mon.nc")


def nse(o, p):
    return 1 - np.sum((o-p)**2) / np.sum((o-o.mean())**2)


def fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=3, random_state=0).fit(scaler.transform(Xtr), ytr)
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    rf.fit(Xtr, ytr)
    return en.predict(scaler.transform(Xte)), rf.predict(Xte)


all_rows = []
for name, cfg in BASINS.items():
    print(f"\n=== {name} ===")
    lon, lat = cfg["box_center"]
    half = cfg["half"]

    sst_url = (f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv?"
               f"sst[(2003-01-01):1:(2026-06-30)][(0.0)][({lat-half}):({lat+half})][({360+lon-half}):({360+lon+half})]")
    result = subprocess.run(["curl", "-sgL", sst_url], capture_output=True, text=True, timeout=240)
    lines = result.stdout.splitlines()
    sst = pd.read_csv(pd.io.common.StringIO(result.stdout), skiprows=[1])
    sst = sst.dropna(subset=["sst"])
    sst["date"] = pd.to_datetime(sst["time"])
    sst["ym"] = sst["date"].dt.to_period("M")
    sst_monthly = sst.groupby("ym")["sst"].mean().to_frame("sst_mean")
    clim = sst_monthly.groupby(sst_monthly.index.month)["sst_mean"].transform("mean")
    sst_monthly["sst_anomaly"] = sst_monthly["sst_mean"] - clim
    print(f"  SST: {len(sst_monthly)} months")

    lon360 = 360 + lon
    u_box = u["uwnd"].sel(lat=lat, lon=lon360, method="nearest")
    v_box = v["vwnd"].sel(lat=lat, lon=lon360, method="nearest")
    speed = np.sqrt(u_box**2 + v_box**2)
    wind_df = speed.to_dataframe(name="wind_speed").reset_index()[["time", "wind_speed"]]
    wind_df["ym"] = pd.PeriodIndex(pd.to_datetime(wind_df["time"]).dt.to_period("M"), freq="M")
    wind_df = wind_df.groupby("ym")["wind_speed"].mean().to_frame()
    clim_w = wind_df.groupby(wind_df.index.month)["wind_speed"].transform("mean")
    wind_df["wind_speed_anomaly"] = wind_df["wind_speed"] - clim_w

    base = pd.read_csv(PROC_W / f"{name}_stage_b_monthly.csv")
    base["ym"] = pd.PeriodIndex(base["ym"], freq="M")
    df = base.merge(sst_monthly[["sst_anomaly"]], left_on="ym", right_index=True, how="inner")
    df = df.merge(wind_df[["wind_speed_anomaly"]], left_on="ym", right_index=True, how="inner")
    features_new = FEATURES_BASE + ["sst_anomaly", "wind_speed_anomaly"]
    df = df.dropna(subset=features_new + [TARGET]).reset_index(drop=True)
    if len(df) < 40:
        print(f"  too few rows ({len(df)}) after SST/wind join -- skipping")
        continue

    y = df[TARGET].values
    for feat_label, feats in [("base", FEATURES_BASE), ("with_sst_wind", features_new)]:
        X = df[feats].values
        for scheme_name, splitter in [("expanding", TimeSeriesSplit(n_splits=5)), ("blocked", KFold(n_splits=5, shuffle=False))]:
            for i, (tr, te) in enumerate(splitter.split(X)):
                en_pred, rf_pred = fit_predict(X[tr], y[tr], X[te])
                all_rows.append({"basin": name, "feature_set": feat_label, "scheme": scheme_name, "fold": i+1,
                                  "NSE_clim": nse(y[te], np.zeros_like(y[te])),
                                  "NSE_EN": nse(y[te], en_pred), "NSE_RF": nse(y[te], rf_pred)})
    summary = pd.DataFrame(all_rows)
    summary = summary[summary["basin"] == name].groupby(["feature_set", "scheme"])[["NSE_EN"]].mean()
    print(summary)

results = pd.DataFrame(all_rows)
results.to_csv(PROC_W / "sst_wind_more_basins_cv_scores.csv", index=False)
print("\n\n=== SUMMARY: base vs with_sst_wind, mean NSE_EN by basin+scheme ===")
print(results.groupby(["basin", "feature_set", "scheme"])["NSE_EN"].mean().unstack("feature_set").to_string())
