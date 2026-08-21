"""
Stage B, next iteration: add SST and wind-speed anomalies as covariates to
the far_offshore model, per the "Next steps #2" item in results_summary.md
-- coastal chlorophyll is also driven by wind-driven mixing/upwelling and
thermal stratification, not discharge alone, so the model was likely
misattributing some of that variance as unexplained noise.

Sources:
- SST: NOAA OISST v2.1 daily, 0.25 deg (ERDDAP ncdcOisst21Agg)
- Wind: NCEP/NCAR Reanalysis monthly 10m u/v wind (NOAA PSL, direct NetCDF,
  downloaded to data/raw/{u,v}wnd_10m_mon.nc), ~1.9 deg resolution -- coarse
  relative to our other sources, but reanalysis wind fields are inherently
  smooth/large-scale, so this resolution is appropriate for the phenomenon.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import shap

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

BOX = dict(lon_min=-65.93, lon_max=-65.83, lat_min=18.46, lat_max=18.54)

# ---- SST: daily -> monthly anomaly ----
sst = pd.read_csv(RAW / "oisst_far_offshore_daily.csv", skiprows=[1])
sst = sst.dropna(subset=["sst"])
sst["date"] = pd.to_datetime(sst["time"])
sst["ym"] = sst["date"].dt.to_period("M")
sst_monthly = sst.groupby("ym")["sst"].mean().rename("sst_mean").to_frame()
clim = sst_monthly.groupby(sst_monthly.index.month)["sst_mean"].transform("mean")
sst_monthly["sst_anomaly"] = sst_monthly["sst_mean"] - clim
print(f"SST monthly: {len(sst_monthly)} months, {sst_monthly.index.min()} to {sst_monthly.index.max()}")

# ---- Wind: NCEP/NCAR reanalysis monthly u/v -> speed anomaly, box-mean ----
u = xr.open_dataset(RAW / "uwnd_10m_mon.nc")
v = xr.open_dataset(RAW / "vwnd_10m_mon.nc")
# Reanalysis grid is ~1.9deg resolution -- much coarser than our ~0.1deg box,
# so a range slice misses every grid point. Use the nearest single cell to
# the box center instead (appropriate: reanalysis wind fields are smooth/
# large-scale by construction, a coarse cell is representative here).
box_center_lon360 = 360 + (BOX["lon_min"] + BOX["lon_max"]) / 2
box_center_lat = (BOX["lat_min"] + BOX["lat_max"]) / 2
u_box = u["uwnd"].sel(lat=box_center_lat, lon=box_center_lon360, method="nearest")
v_box = v["vwnd"].sel(lat=box_center_lat, lon=box_center_lon360, method="nearest")
print(f"Nearest wind grid cell: lat={float(u_box.lat):.2f}, lon={float(u_box.lon):.2f}")
speed = np.sqrt(u_box ** 2 + v_box ** 2)
wind_df = speed.to_dataframe(name="wind_speed").reset_index()[["time", "wind_speed"]]
wind_df["ym"] = pd.PeriodIndex(pd.to_datetime(wind_df["time"]).dt.to_period("M"), freq="M")
wind_df = wind_df.groupby("ym")["wind_speed"].mean().to_frame()
clim_w = wind_df.groupby(wind_df.index.month)["wind_speed"].transform("mean")
wind_df["wind_speed_anomaly"] = wind_df["wind_speed"] - clim_w
print(f"Wind monthly: {len(wind_df)} months, {wind_df.index.min()} to {wind_df.index.max()}")
print(f"Box mean grid cells used: u_box shape {u_box.shape}, lat {u_box.lat.values}, lon {u_box.lon.values}")

# ---- Join onto the existing far_offshore Stage B dataset ----
base = pd.read_csv(PROC / "far_offshore_stage_b_monthly.csv")
base["ym"] = pd.PeriodIndex(base["ym"], freq="M")
df = base.merge(sst_monthly[["sst_anomaly"]], left_on="ym", right_index=True, how="inner")
df = df.merge(wind_df[["wind_speed_anomaly"]], left_on="ym", right_index=True, how="inner")

FEATURES_BASE = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
                  "spi_3", "month_sin", "month_cos"]
FEATURES_NEW = FEATURES_BASE + ["sst_anomaly", "wind_speed_anomaly"]
TARGET = "log_chl_anomaly"
df = df.dropna(subset=FEATURES_NEW + [TARGET]).reset_index(drop=True)
df.to_csv(PROC / "far_offshore_stage_b_with_sst_wind.csv", index=False)
print(f"\nUsable rows with SST+wind: {len(df)} ({df['ym'].iloc[0]} to {df['ym'].iloc[-1]})")


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


def run_cv(X, y, label):
    print(f"\n=== {label} ===")
    rows = []
    for scheme_name, splitter in [("expanding", TimeSeriesSplit(n_splits=5)), ("blocked", KFold(n_splits=5, shuffle=False))]:
        for i, (tr, te) in enumerate(splitter.split(X)):
            en_pred, rf_pred, _ = fit_predict(X[tr], y[tr], X[te])
            rows.append({"scheme": scheme_name, "fold": i + 1,
                         "NSE_clim": nse(y[te], np.zeros_like(y[te])),
                         "NSE_EN": nse(y[te], en_pred), "NSE_RF": nse(y[te], rf_pred)})
    res = pd.DataFrame(rows)
    summary = res.groupby("scheme")[["NSE_clim", "NSE_EN", "NSE_RF"]].agg(["mean", "std"])
    print(summary)
    return res


y = df[TARGET].values
res_base = run_cv(df[FEATURES_BASE].values, y, "Baseline features only (discharge+SPI+season)")
res_new = run_cv(df[FEATURES_NEW].values, y, "With SST + wind added")

pd.concat([res_base.assign(feature_set="base"), res_new.assign(feature_set="with_sst_wind")]).to_csv(
    PROC / "far_offshore_sst_wind_cv_scores.csv", index=False)

# ---- Final holdout comparison + SHAP on the richer feature set ----
split = int(len(df) * 0.8)
X_new = df[FEATURES_NEW].values
Xtr, Xte, ytr, yte = X_new[:split], X_new[split:], y[:split], y[split:]
en_pred, rf_pred, rf_model = fit_predict(Xtr, ytr, Xte)
rho, _ = spearmanr(yte, en_pred)
print(f"\n=== Final holdout, WITH SST+wind ({len(yte)} months) ===")
print(f"ElasticNet: NSE={nse(yte, en_pred):+.3f}  Spearman={rho:+.3f}")
print(f"RandomForest: NSE={nse(yte, rf_pred):+.3f}")

rng = np.random.default_rng(0)
block = 6
n_blocks = len(yte) // block
boot_nse = []
for _ in range(1000):
    idx = rng.integers(0, n_blocks, n_blocks)
    obs_bs = np.concatenate([yte[b*block:(b+1)*block] for b in idx])
    pred_bs = np.concatenate([en_pred[b*block:(b+1)*block] for b in idx])
    boot_nse.append(nse(obs_bs, pred_bs))
ci_lo, ci_hi = np.percentile(boot_nse, [2.5, 97.5])
print(f"ElasticNet NSE 95% CI (with SST+wind): [{ci_lo:.3f}, {ci_hi:.3f}]")

FEATURE_LABELS = {
    "log_q_anomaly_lag0": "Discharge anomaly, t", "log_q_anomaly_lag1": "Discharge anomaly, t-1",
    "log_q_anomaly_lag2": "Discharge anomaly, t-2", "log_q_anomaly_lag3": "Discharge anomaly, t-3",
    "spi_3": "SPI-3", "month_sin": "Seasonal phase (sin)", "month_cos": "Seasonal phase (cos)",
    "sst_anomaly": "SST anomaly", "wind_speed_anomaly": "Wind speed anomaly",
}
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(Xte)
fig = plt.figure(figsize=(8, 5.5))
shap.summary_plot(shap_values, Xte, feature_names=[FEATURE_LABELS[f] for f in FEATURES_NEW],
                   plot_type="bar", show=False, color="#7c3aed")
plt.title("SHAP feature importance -- with SST + wind covariates\n(RandomForest, final holdout)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(FIG / "10_shap_with_sst_wind.png")
plt.close(fig)
print(f"\nSaved SHAP plot to {FIG / '10_shap_with_sst_wind.png'}")
