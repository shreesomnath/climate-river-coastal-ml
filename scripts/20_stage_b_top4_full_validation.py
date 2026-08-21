"""
Full Stage B validation (MODIS+VIIRS merge, two folding schemes, baselines,
final holdout + bootstrap CI, SHAP) for the 4 strongest candidates from the
coastal-box screen: Manati, Plata, Anasco (offshore box wins), Culebrinas
(near_mouth box wins, barely). Same rigor as Loiza's far_offshore pipeline
(script 12), generalized across basins.
"""
import subprocess
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
import shap

ROOT = Path(__file__).resolve().parent.parent
PROC_W = ROOT / "data" / "processed" / "watersheds"
FIG = ROOT / "figures" / "watersheds"
FIG.mkdir(parents=True, exist_ok=True)

BASINS = {
    "fajardo":   dict(box_center=(-65.538, 18.339), half=0.05),  # offshore won the screen
    "patillas":  dict(box_center=(-65.934, 17.939), half=0.05),  # offshore won the screen
    "guanajibo": dict(box_center=(-67.181, 18.168), half=0.05),  # near_mouth won the screen
}


def fetch_box(dataset, var, time_range, box, extra_dim=""):
    url = (f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{dataset}.csv?"
           f"{var}[{time_range}]{extra_dim}[({box['lat_min']}):({box['lat_max']})][({box['lon_min']}):({box['lon_max']})]")
    result = subprocess.run(["curl", "-sgL", url], capture_output=True, text=True, timeout=180)
    reader = csv.reader(result.stdout.splitlines()[2:])
    by_month = defaultdict(list)
    ncol = 5 if extra_dim else 4
    for row in reader:
        if len(row) < ncol or row[-1] == "NaN":
            continue
        by_month[row[0][:7]].append(float(row[-1]))
    return by_month


def nse(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)


def fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=3, random_state=0).fit(scaler.transform(Xtr), ytr)
    en_pred = en.predict(scaler.transform(Xte))
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    return en_pred, rf_pred, rf


FEATURES = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
            "spi_3", "month_sin", "month_cos"]
TARGET = "log_chl_anomaly"

all_results = []
for name, cfg in BASINS.items():
    print(f"\n{'='*20} {name} {'='*20}")
    lon, lat = cfg["box_center"]
    half = cfg["half"]
    box = dict(lon_min=lon - half, lon_max=lon + half, lat_min=lat - half, lat_max=lat + half)

    modis = fetch_box("erdMH1chlamday", "chlorophyll", "(2003-01-01):1:(2022-03-16)", box)
    viirs = fetch_box("nesdisVHNSQchlaMonthly", "chlor_a", "(2012-01-01):1:(2026-06-01)", box, extra_dim="[(0.0)]")
    print(f"  MODIS: {len(modis)} months, VIIRS: {len(viirs)} months")
    if len(modis) < 50 or len(viirs) < 50:
        print("  INSUFFICIENT DATA -- skipping")
        continue

    modis_df = pd.DataFrame([(k, np.mean(v)) for k, v in modis.items()], columns=["ym", "chl"])
    viirs_df = pd.DataFrame([(k, np.mean(v)) for k, v in viirs.items()], columns=["ym", "chl"])
    modis_df["log_chl"] = np.log(modis_df["chl"])
    viirs_df["log_chl"] = np.log(viirs_df["chl"])
    overlap = modis_df[["ym", "log_chl"]].merge(viirs_df[["ym", "log_chl"]], on="ym", suffixes=("_m", "_v"))
    if len(overlap) < 20:
        print(f"  Only {len(overlap)} overlap months -- insufficient for bias correction, skipping")
        continue
    reg = LinearRegression().fit(overlap[["log_chl_v"]], overlap["log_chl_m"])
    r2 = reg.score(overlap[["log_chl_v"]], overlap["log_chl_m"])
    print(f"  Bias correction R^2={r2:.3f} (n_overlap={len(overlap)})")
    viirs_df["log_chl_corr"] = reg.predict(viirs_df[["log_chl"]].rename(columns={"log_chl": "log_chl_v"}))
    modis_part = modis_df[["ym", "log_chl"]].rename(columns={"log_chl": "log_chl_merged"})
    viirs_post = viirs_df[viirs_df["ym"] > modis_df["ym"].max()][["ym", "log_chl_corr"]].rename(
        columns={"log_chl_corr": "log_chl_merged"})
    chl = pd.concat([modis_part, viirs_post], ignore_index=True).sort_values("ym")
    chl["ym"] = pd.PeriodIndex(chl["ym"], freq="M")
    clim = chl.groupby(chl["ym"].dt.month)["log_chl_merged"].transform("mean")
    chl["log_chl_anomaly"] = chl["log_chl_merged"] - clim

    stage_a = pd.read_csv(PROC_W / f"{name}_stage_a_monthly.csv")
    stage_a["ym"] = pd.PeriodIndex(stage_a[stage_a.columns[0]], freq="M")
    df = stage_a.merge(chl[["ym", "log_chl_anomaly"]], on="ym", how="inner").sort_values("ym").reset_index(drop=True)
    for lag in (0, 1, 2, 3):
        df[f"log_q_anomaly_lag{lag}"] = df["log_q_anomaly"].shift(lag) if lag > 0 else df["log_q_anomaly"]
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"  Stage B usable rows: {len(df)} ({df['ym'].iloc[0] if len(df) else '-'} to {df['ym'].iloc[-1] if len(df) else '-'})")
    if len(df) < 60:
        print("  TOO FEW ROWS after merge -- skipping model fit")
        continue

    X, y = df[FEATURES].values, df[TARGET].values
    fold_rows = []
    for scheme_name, splitter in [("expanding", TimeSeriesSplit(n_splits=5)), ("blocked", KFold(n_splits=5, shuffle=False))]:
        for i, (tr, te) in enumerate(splitter.split(X)):
            en_pred, rf_pred, _ = fit_predict(X[tr], y[tr], X[te])
            fold_rows.append({"basin": name, "scheme": scheme_name, "fold": i + 1,
                               "NSE_clim": nse(y[te], np.zeros_like(y[te])),
                               "NSE_EN": nse(y[te], en_pred), "NSE_RF": nse(y[te], rf_pred)})
    fold_df = pd.DataFrame(fold_rows)
    summary = fold_df.groupby("scheme")[["NSE_clim", "NSE_EN", "NSE_RF"]].mean()
    print(summary)
    all_results.append(fold_df)

    # final holdout + bootstrap
    split = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    en_pred, rf_pred, rf_model = fit_predict(Xtr, ytr, Xte)
    rng = np.random.default_rng(0)
    block = 6
    n_blocks = max(1, len(yte) // block)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, n_blocks, n_blocks)
        obs_bs = np.concatenate([yte[b*block:(b+1)*block] for b in idx])
        pred_bs = np.concatenate([en_pred[b*block:(b+1)*block] for b in idx])
        if len(obs_bs) > 1:
            boot.append(nse(obs_bs, pred_bs))
    ci = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    print(f"  Final holdout ({len(yte)} mo): NSE_EN={nse(yte,en_pred):+.3f}, "
          f"NSE_RF={nse(yte,rf_pred):+.3f}, EN 95% CI=[{ci[0]:.3f},{ci[1]:.3f}]")

results_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
out_path = PROC_W / "stage_b_remaining3_cv_scores.csv"
results_all.to_csv(out_path, index=False)
print("\n\nSaved combined results to", out_path)
