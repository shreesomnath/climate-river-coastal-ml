"""
Fills the metrics gap flagged after the top4+remaining3 runs: those only
computed NSE. This adds the full suite (NSE, KGE, RMSE, MAE, Spearman) on
the final holdout for every testable basin (6 of 7 -- Fajardo is excluded,
genuinely untestable: its usable discharge record ends 1996, chlorophyll
only starts 2003, zero overlap), matching the depth Loiza's far_offshore
result got. Also saves each basin's merged Stage B monthly dataset to CSV
(not done before -- re-fetches were required to get here) so this doesn't
need to be repeated again.
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
PROC_W = ROOT / "data" / "processed" / "watersheds"
FIG = ROOT / "figures"

BASINS = {
    "manati":    dict(box_center=(-66.551, 18.569), half=0.05),
    "plata":     dict(box_center=(-66.196, 18.544), half=0.05),
    "anasco":    dict(box_center=(-67.278, 18.272), half=0.05),
    "culebrinas":dict(box_center=(-67.177, 18.406), half=0.05),
    "patillas":  dict(box_center=(-65.934, 17.939), half=0.05),
    "guanajibo": dict(box_center=(-67.181, 18.168), half=0.05),
}
FEATURES = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
            "spi_3", "month_sin", "month_cos"]


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


def kge(obs, pred):
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = pred.std() / obs.std()
    beta = pred.mean() / obs.mean() if obs.mean() != 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae(obs, pred):
    return float(np.mean(np.abs(obs - pred)))


rows = []
for name, cfg in BASINS.items():
    print(f"\n=== {name} ===")
    lon, lat = cfg["box_center"]
    half = cfg["half"]
    box = dict(lon_min=lon - half, lon_max=lon + half, lat_min=lat - half, lat_max=lat + half)

    modis = fetch_box("erdMH1chlamday", "chlorophyll", "(2003-01-01):1:(2022-03-16)", box)
    viirs = fetch_box("nesdisVHNSQchlaMonthly", "chlor_a", "(2012-01-01):1:(2026-06-01)", box, extra_dim="[(0.0)]")
    modis_df = pd.DataFrame([(k, np.mean(v)) for k, v in modis.items()], columns=["ym", "chl"])
    viirs_df = pd.DataFrame([(k, np.mean(v)) for k, v in viirs.items()], columns=["ym", "chl"])
    modis_df["log_chl"] = np.log(modis_df["chl"])
    viirs_df["log_chl"] = np.log(viirs_df["chl"])
    overlap = modis_df[["ym", "log_chl"]].merge(viirs_df[["ym", "log_chl"]], on="ym", suffixes=("_m", "_v"))
    reg = LinearRegression().fit(overlap[["log_chl_v"]], overlap["log_chl_m"])
    viirs_df["log_chl_corr"] = reg.predict(viirs_df[["log_chl"]].rename(columns={"log_chl": "log_chl_v"}))
    modis_part = modis_df[["ym", "log_chl"]].rename(columns={"log_chl": "log_chl_merged"})
    viirs_post = viirs_df[viirs_df["ym"] > modis_df["ym"].max()][["ym", "log_chl_corr"]].rename(columns={"log_chl_corr": "log_chl_merged"})
    chl = pd.concat([modis_part, viirs_post], ignore_index=True).sort_values("ym")
    chl["ym"] = pd.PeriodIndex(chl["ym"], freq="M")
    clim = chl.groupby(chl["ym"].dt.month)["log_chl_merged"].transform("mean")
    chl["log_chl_anomaly"] = chl["log_chl_merged"] - clim

    stage_a = pd.read_csv(PROC_W / f"{name}_stage_a_monthly.csv")
    stage_a["ym"] = pd.PeriodIndex(stage_a[stage_a.columns[0]], freq="M")
    df = stage_a.merge(chl[["ym", "log_chl_anomaly"]], on="ym", how="inner").sort_values("ym").reset_index(drop=True)
    for lag in (0, 1, 2, 3):
        df[f"log_q_anomaly_lag{lag}"] = df["log_q_anomaly"].shift(lag) if lag > 0 else df["log_q_anomaly"]
    df = df.dropna(subset=FEATURES + ["log_chl_anomaly"]).reset_index(drop=True)
    df.to_csv(PROC_W / f"{name}_stage_b_monthly.csv", index=False)  # save for reuse

    X, y = df[FEATURES].values, df["log_chl_anomaly"].values
    split = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    persist_pred = df["log_chl_anomaly"].shift(1).values[split:]
    valid_p = ~np.isnan(persist_pred)

    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=3, random_state=0).fit(scaler.transform(Xtr), ytr)
    en_pred = en.predict(scaler.transform(Xte))
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)

    for model_name, pred, obs_full in [
        ("Climatology", np.zeros_like(yte), yte),
        ("Persistence", persist_pred[valid_p], yte[valid_p]),
        ("ElasticNet", en_pred, yte),
        ("RandomForest", rf_pred, yte),
    ]:
        p = pred if model_name != "Persistence" else pred
        o = obs_full
        rho = spearmanr(o, p)[0] if np.std(p) > 1e-9 else np.nan
        kge_val = kge(o, p) if np.std(p) > 1e-9 else np.nan
        rows.append({
            "basin": name, "model": model_name, "n_test": len(o),
            "NSE": round(nse(o, p), 3), "KGE": round(kge_val, 3) if not np.isnan(kge_val) else np.nan,
            "RMSE": round(rmse(o, p), 3), "MAE": round(mae(o, p), 3),
            "Spearman": round(rho, 3) if not np.isnan(rho) else np.nan,
        })
    print(f"  n_test={len(yte)}, ElasticNet NSE={nse(yte,en_pred):+.3f}")

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(PROC_W / "stage_b_full_metrics_all_basins.csv", index=False)
print("\n\n=== FULL METRIC SUITE, ALL TESTABLE BASINS ===")
print(metrics_df.to_string(index=False))

# ============ Plot: NSE + KGE + RMSE side by side, ElasticNet only, all basins ============
en_only = metrics_df[metrics_df["model"] == "ElasticNet"].copy()
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
colors = ["#0f766e" if v > 0 else "#c1121f" for v in en_only["NSE"]]
for ax, metric, better in zip(axes, ["NSE", "KGE", "RMSE"], ["higher", "higher", "lower"]):
    ax.barh(en_only["basin"], en_only[metric], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} ({better} = better)")
fig.suptitle("ElasticNet final-holdout metrics, all testable basins", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "16_all_basins_full_metrics.png")
plt.close(fig)
print(f"\nSaved {FIG / '16_all_basins_full_metrics.png'}")
