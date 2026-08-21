"""
Final synthesis for the multi-basin Stage B expansion: consolidate all CV
results (top4 + remaining3) into one table, build the comparison figure,
and regenerate Manati's SHAP plot (the second confirmed basin) for driver
attribution -- same treatment Loiza's far_offshore result got.
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
import shap

ROOT = Path(__file__).resolve().parent.parent
PROC_W = ROOT / "data" / "processed" / "watersheds"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# ============ 1. Consolidate all CV results ============
top4 = pd.read_csv(PROC_W / "stage_b_top4_cv_scores.csv")
rem3 = pd.read_csv(PROC_W / "stage_b_remaining3_cv_scores.csv")
combined = pd.concat([top4, rem3], ignore_index=True)
combined.to_csv(PROC_W / "stage_b_all_basins_cv_scores.csv", index=False)

summary = combined.groupby(["basin", "scheme"])[["NSE_clim", "NSE_EN", "NSE_RF"]].mean().reset_index()
print("=== Per-basin, per-scheme mean NSE ===")
print(summary.to_string(index=False))

BASIN_NAMES = {
    "loiza": "Loíza", "manati": "Manatí", "plata": "Plata", "anasco": "Añasco",
    "culebrinas": "Culebrinas", "patillas": "Patillas", "guanajibo": "Guanajibo", "fajardo": "Fajardo",
}

# ============ 2. Comparison figure: NSE_EN by basin, both schemes, incl. Loiza ============
loiza_row = pd.DataFrame([
    {"basin": "loiza", "scheme": "expanding", "NSE_EN": 0.082},
    {"basin": "loiza", "scheme": "blocked", "NSE_EN": 0.144},
])
plot_df = pd.concat([summary[["basin", "scheme", "NSE_EN"]], loiza_row], ignore_index=True)
basins_order = ["loiza", "manati", "patillas", "plata", "anasco", "culebrinas", "guanajibo"]
plot_df["basin"] = pd.Categorical(plot_df["basin"], categories=basins_order, ordered=True)
plot_df = plot_df.sort_values("basin")

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(basins_order))
width = 0.35
exp_vals = [plot_df[(plot_df.basin == b) & (plot_df.scheme == "expanding")]["NSE_EN"].values for b in basins_order]
blk_vals = [plot_df[(plot_df.basin == b) & (plot_df.scheme == "blocked")]["NSE_EN"].values for b in basins_order]
exp_vals = [v[0] if len(v) else np.nan for v in exp_vals]
blk_vals = [v[0] if len(v) else np.nan for v in blk_vals]
ax.bar(x - width/2, exp_vals, width, label="Expanding-window CV", color="#1f5fa8")
ax.bar(x + width/2, blk_vals, width, label="Blocked K-fold CV", color="#0f766e")
ax.axhline(0, color="black", lw=1.0)
ax.set_xticks(x)
labels = [BASIN_NAMES[b] for b in basins_order]
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("NSE (ElasticNet)")
ax.set_title("Stage B (discharge → coastal chlorophyll): all 7 tested basins")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "14_multibasin_stage_b_comparison.png")
plt.close(fig)
print(f"\nSaved {FIG / '14_multibasin_stage_b_comparison.png'}")

# ============ 3. Manati SHAP (re-fit quickly on its saved Stage B data) ============
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

box = dict(lon_min=-66.601, lon_max=-66.501, lat_min=18.519, lat_max=18.619)
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

stage_a = pd.read_csv(PROC_W / "manati_stage_a_monthly.csv")
stage_a["ym"] = pd.PeriodIndex(stage_a[stage_a.columns[0]], freq="M")
df = stage_a.merge(chl[["ym", "log_chl_anomaly"]], on="ym", how="inner").sort_values("ym").reset_index(drop=True)
FEATURES = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
            "spi_3", "month_sin", "month_cos"]
for lag in (0, 1, 2, 3):
    df[f"log_q_anomaly_lag{lag}"] = df["log_q_anomaly"].shift(lag) if lag > 0 else df["log_q_anomaly"]
df = df.dropna(subset=FEATURES + ["log_chl_anomaly"]).reset_index(drop=True)

X, y = df[FEATURES].values, df["log_chl_anomaly"].values
split = int(len(df) * 0.8)
rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
rf.fit(X[:split], y[:split])
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X[split:])
FEATURE_LABELS = {
    "log_q_anomaly_lag0": "Discharge anomaly, t", "log_q_anomaly_lag1": "Discharge anomaly, t-1",
    "log_q_anomaly_lag2": "Discharge anomaly, t-2", "log_q_anomaly_lag3": "Discharge anomaly, t-3",
    "spi_3": "SPI-3", "month_sin": "Seasonal phase (sin)", "month_cos": "Seasonal phase (cos)",
}
fig = plt.figure(figsize=(8, 5.5))
shap.summary_plot(shap_values, X[split:], feature_names=[FEATURE_LABELS[f] for f in FEATURES],
                   plot_type="bar", show=False, color="#c1121f")
plt.title("SHAP feature importance -- Manatí (2nd confirmed basin)\n(RandomForest, final holdout)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(FIG / "15_shap_manati.png")
plt.close(fig)
print(f"Saved {FIG / '15_shap_manati.png'}")
