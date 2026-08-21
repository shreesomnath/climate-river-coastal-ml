"""
Full Stage B pipeline (fetch -> merge -> preprocess -> model) for the
'far_offshore' coastal box (lon -65.93/-65.83, lat 18.46/18.54), which the
quick correlation screen in 11_test_coastal_boxes.py flagged as far
stronger (|r|=0.501) than the original box (|r|~0.23) or near-mouth
candidates. This script gives it the same full validation rigor as
everything else: MODIS+VIIRS merge for a longer record, baselines,
multiple models, TWO folding strategies (expanding-window AND contiguous
blocked k-fold, per feedback that a single CV scheme isn't enough),
NSE + KGE + RMSE + MAE + Spearman, bootstrap CI on NSE, and SHAP.
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

# ============ 1. Fetch MODIS + VIIRS for this box ============

def fetch_box(dataset, var, time_range, box, extra_dim=""):
    url = (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{dataset}.csv?"
        f"{var}[{time_range}]{extra_dim}"
        f"[({box['lat_min']}):({box['lat_max']})][({box['lon_min']}):({box['lon_max']})]"
    )
    print(f"  fetching: {url}")
    result = subprocess.run(["curl", "-sgL", url], capture_output=True, text=True, check=True, timeout=180)
    lines = result.stdout.splitlines()
    reader = csv.reader(lines[2:])
    by_month = defaultdict(list)
    ncol_expected = 5 if extra_dim else 4
    for row in reader:
        if len(row) < ncol_expected:
            continue
        time_str, val = row[0], row[-1]
        if val == "NaN":
            continue
        by_month[time_str[:7]].append(float(val))
    return by_month


print("Fetching MODIS (2003-2022) for far_offshore box...")
modis_by_month = fetch_box("erdMH1chlamday", "chlorophyll", "(2003-01-01):1:(2022-03-16)", BOX)
print(f"  {len(modis_by_month)} months")

print("Fetching VIIRS (2012-present) for far_offshore box...")
viirs_by_month = fetch_box("nesdisVHNSQchlaMonthly", "chlor_a", "(2012-01-01):1:(2026-06-01)", BOX, extra_dim="[(0.0)]")
print(f"  {len(viirs_by_month)} months")

modis = pd.DataFrame([(k, np.mean(v)) for k, v in modis_by_month.items()], columns=["year_month", "chl"])
viirs = pd.DataFrame([(k, np.mean(v)) for k, v in viirs_by_month.items()], columns=["year_month", "chl"])
modis["log_chl"] = np.log(modis["chl"])
viirs["log_chl"] = np.log(viirs["chl"])

# ============ 2. Bias-correct + splice (same method as 09_merge_chlorophyll_sensors.py) ============
from sklearn.linear_model import LinearRegression

overlap = modis[["year_month", "log_chl"]].merge(viirs[["year_month", "log_chl"]], on="year_month",
                                                    suffixes=("_modis", "_viirs"))
reg = LinearRegression().fit(overlap[["log_chl_viirs"]], overlap["log_chl_modis"])
r2 = reg.score(overlap[["log_chl_viirs"]], overlap["log_chl_modis"])
print(f"Bias correction: n_overlap={len(overlap)}, R^2={r2:.3f}")

viirs["log_chl_corrected"] = reg.predict(viirs[["log_chl"]].rename(columns={"log_chl": "log_chl_viirs"}))
modis_part = modis[["year_month", "log_chl"]].rename(columns={"log_chl": "log_chl_merged"})
viirs_post = viirs[viirs["year_month"] > modis["year_month"].max()][["year_month", "log_chl_corrected"]].rename(
    columns={"log_chl_corrected": "log_chl_merged"})
merged_chl = pd.concat([modis_part, viirs_post], ignore_index=True).sort_values("year_month")
merged_chl.to_csv(PROC / "far_offshore_chlorophyll_merged.csv", index=False)
print(f"Merged chlorophyll series: {len(merged_chl)} months, {merged_chl['year_month'].min()} to {merged_chl['year_month'].max()}")

# ============ 3. Preprocess (join with Stage A discharge/SPI) ============
stage_a = pd.read_csv(PROC / "stage_a_monthly.csv")
stage_a["ym"] = pd.PeriodIndex(stage_a["ym"], freq="M")

chl = merged_chl.copy()
chl["ym"] = pd.PeriodIndex(chl["year_month"], freq="M")
clim = chl.groupby(chl["ym"].dt.month)["log_chl_merged"].transform("mean")
chl["log_chl_anomaly"] = chl["log_chl_merged"] - clim

df = stage_a.merge(chl[["ym", "log_chl_anomaly"]], on="ym", how="inner").sort_values("ym").reset_index(drop=True)
for lag in (0, 1, 2, 3):
    df[f"log_q_anomaly_lag{lag}"] = df["log_q_anomaly"].shift(lag) if lag > 0 else df["log_q_anomaly"]

FEATURES = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
            "spi_3", "month_sin", "month_cos"]
TARGET = "log_chl_anomaly"
df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
df.to_csv(PROC / "far_offshore_stage_b_monthly.csv", index=False)
print(f"\nStage B (far_offshore) usable rows: {len(df)} ({df['ym'].iloc[0]} to {df['ym'].iloc[-1]})")

X, y = df[FEATURES].values, df[TARGET].values

# ============ 4. Metrics ============

def nse(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)

def kge(obs, pred):
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = pred.std() / obs.std()
    beta = pred.mean() / obs.mean() if obs.mean() != 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), r, alpha, beta

def rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))

def mae(obs, pred):
    return float(np.mean(np.abs(obs - pred)))


def fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=3, random_state=0).fit(scaler.transform(Xtr), ytr)
    en_pred = en.predict(scaler.transform(Xte))
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    return en_pred, rf_pred, rf


# ============ 5a. Expanding-window TimeSeriesSplit CV ============
print("\n=== Folding A: expanding-window TimeSeriesSplit (5 folds) ===")
tscv = TimeSeriesSplit(n_splits=5)
rows_a = []
for i, (tr, te) in enumerate(tscv.split(X)):
    en_pred, rf_pred, _ = fit_predict(X[tr], y[tr], X[te])
    clim_pred = np.zeros_like(y[te])
    rows_a.append({
        "fold": i + 1, "n_test": len(te),
        "NSE_clim": round(nse(y[te], clim_pred), 3),
        "NSE_EN": round(nse(y[te], en_pred), 3),
        "NSE_RF": round(nse(y[te], rf_pred), 3),
        "KGE_EN": round(kge(y[te], en_pred)[0], 3),
        "KGE_RF": round(kge(y[te], rf_pred)[0], 3),
    })
res_a = pd.DataFrame(rows_a)
print(res_a.to_string(index=False))
print("Mean +/- SD:", {c: f"{res_a[c].mean():.3f}+/-{res_a[c].std():.3f}" for c in res_a.columns if c not in ("fold", "n_test")})

# ============ 5b. Contiguous blocked K-Fold (non-expanding, per feedback: try more than one folding scheme) ============
print("\n=== Folding B: contiguous blocked K-Fold (5 folds, shuffle=False) ===")
kf = KFold(n_splits=5, shuffle=False)
rows_b = []
for i, (tr, te) in enumerate(kf.split(X)):
    en_pred, rf_pred, _ = fit_predict(X[tr], y[tr], X[te])
    clim_pred = np.zeros_like(y[te])
    rows_b.append({
        "fold": i + 1, "n_test": len(te),
        "NSE_clim": round(nse(y[te], clim_pred), 3),
        "NSE_EN": round(nse(y[te], en_pred), 3),
        "NSE_RF": round(nse(y[te], rf_pred), 3),
        "KGE_EN": round(kge(y[te], en_pred)[0], 3),
        "KGE_RF": round(kge(y[te], rf_pred)[0], 3),
    })
res_b = pd.DataFrame(rows_b)
print(res_b.to_string(index=False))
print("Mean +/- SD:", {c: f"{res_b[c].mean():.3f}+/-{res_b[c].std():.3f}" for c in res_b.columns if c not in ("fold", "n_test")})

pd.concat([res_a.assign(scheme="expanding_window"), res_b.assign(scheme="blocked_kfold")]).to_csv(
    PROC / "far_offshore_cv_scores.csv", index=False)

# ============ 6. Final holdout (last 20%) with full metric suite + bootstrap CI ============
split = int(len(df) * 0.8)
Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
en_pred, rf_pred, rf_model = fit_predict(Xtr, ytr, Xte)
persist_pred = df["log_chl_anomaly"].shift(1).values[split:]
valid_p = ~np.isnan(persist_pred)

print(f"\n=== Final holdout ({len(yte)} months) -- full metric suite ===")
for name, pred in [("Climatology", np.zeros_like(yte)),
                    ("Persistence", persist_pred if valid_p.all() else None),
                    ("ElasticNet", en_pred), ("RandomForest", rf_pred)]:
    if pred is None:
        continue
    obs_ = yte[valid_p] if name == "Persistence" else yte
    p_ = pred[valid_p] if name == "Persistence" else pred
    kge_val, r, alpha, beta = kge(obs_, p_)
    rho, _ = spearmanr(obs_, p_)
    print(f"{name:14s} NSE={nse(obs_,p_):+.3f}  KGE={kge_val:+.3f} (r={r:.2f},a={alpha:.2f},b={beta:.2f})  "
          f"RMSE={rmse(obs_,p_):.3f}  MAE={mae(obs_,p_):.3f}  Spearman={rho:+.3f}")

# Bootstrap CI on ElasticNet NSE (block bootstrap, resampling contiguous 6-month chunks to respect autocorrelation)
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
print(f"\nElasticNet NSE 95% block-bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}] (point est {nse(yte,en_pred):.3f})")

# ============ 7. SHAP ============
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(Xte)
FEATURE_LABELS = {
    "log_q_anomaly_lag0": "Discharge anomaly, t", "log_q_anomaly_lag1": "Discharge anomaly, t-1",
    "log_q_anomaly_lag2": "Discharge anomaly, t-2", "log_q_anomaly_lag3": "Discharge anomaly, t-3",
    "spi_3": "SPI-3", "month_sin": "Seasonal phase (sin)", "month_cos": "Seasonal phase (cos)",
}
fig = plt.figure(figsize=(8, 5.5))
shap.summary_plot(shap_values, Xte, feature_names=[FEATURE_LABELS[f] for f in FEATURES],
                   plot_type="bar", show=False, color="#0f766e")
plt.title("SHAP feature importance -- far_offshore box\n(RandomForest, final holdout)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(FIG / "09_shap_far_offshore.png")
plt.close(fig)
print(f"\nSaved SHAP plot to {FIG / '09_shap_far_offshore.png'}")
