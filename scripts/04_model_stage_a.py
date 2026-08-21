"""
Stage A modeling: Climate (SPI-1/3/6/12, precip, seasonality, persistence) -> log-discharge anomaly.
Implements the protocol in docs/validation_protocol.md and docs/modeling_spec.md:
- chronological train/test split (no shuffling)
- TimeSeriesSplit CV within training period
- baselines: climatology (0) and persistence (lag-1)
- models: ElasticNet, RandomForest, XGBoost
- metrics: NSE, RMSE, MAE on held-out test only
- SHAP attribution on the best tree model
- plots: cross-correlation (lag scan), time series overlay, obs-vs-pred, SHAP summary
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401 -- applies publication rcParams on import
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = ["spi_1", "spi_3", "spi_6", "spi_12", "precip_mm_sum",
            "month_sin", "month_cos", "log_q_anomaly_lag1"]
TARGET = "log_q_anomaly"

df = pd.read_csv(PROC / "stage_a_monthly.csv")
df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
print(f"Usable rows after dropping NaNs: {len(df)} ({df['ym'].iloc[0]} to {df['ym'].iloc[-1]})")

# ---- Chronological split: last 20% held out ----
n = len(df)
split = int(n * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
print(f"Train: {len(train)} rows ({train['ym'].iloc[0]}-{train['ym'].iloc[-1]}) | "
      f"Test: {len(test)} rows ({test['ym'].iloc[0]}-{test['ym'].iloc[-1]})")

X_train, y_train = train[FEATURES].values, train[TARGET].values
X_test, y_test = test[FEATURES].values, test[TARGET].values


def nse(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)


def rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae(obs, pred):
    return float(np.mean(np.abs(obs - pred)))


def score(name, pred):
    return {"model": name, "NSE": round(nse(y_test, pred), 3),
            "RMSE": round(rmse(y_test, pred), 3), "MAE": round(mae(y_test, pred), 3)}


results = []

# ---- Baselines ----
results.append(score("Climatology (predict 0)", np.zeros_like(y_test)))
results.append(score("Persistence (t-1)", test["log_q_anomaly_lag1"].values))

# ---- ElasticNet (standardized) ----
scaler = StandardScaler().fit(X_train)
en = ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), random_state=0).fit(scaler.transform(X_train), y_train)
results.append(score("ElasticNet", en.predict(scaler.transform(X_test))))

# ---- Random Forest ----
rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=4, random_state=0)
rf.fit(X_train, y_train)
results.append(score("RandomForest", rf.predict(X_test)))

# ---- XGBoost ----
xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=0)
xgb.fit(X_train, y_train)
results.append(score("XGBoost", xgb.predict(X_test)))

results_df = pd.DataFrame(results)
print("\n=== Test-set skill vs baselines ===")
print(results_df.to_string(index=False))
results_df.to_csv(PROC / "stage_a_model_scores.csv", index=False)

# ---- TimeSeriesSplit CV spread on training period (RF) ----
tscv = TimeSeriesSplit(n_splits=5)
cv_nse = []
for tr_idx, val_idx in tscv.split(X_train):
    m = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=4, random_state=0)
    m.fit(X_train[tr_idx], y_train[tr_idx])
    cv_nse.append(nse(y_train[val_idx], m.predict(X_train[val_idx])))
print(f"\nRF TimeSeriesSplit CV NSE (train period, 5 folds): "
      f"mean={np.mean(cv_nse):.3f} +/- {np.std(cv_nse):.3f}  (folds: {[round(v,3) for v in cv_nse]})")

# =================== PLOTS ===================

# 1. Time series overlay: discharge anomaly vs SPI-12
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ym = pd.PeriodIndex(df["ym"], freq="M").to_timestamp()
ax1.plot(ym, df["log_q_anomaly"], color="#1f5fa8", lw=1.1, label="Log discharge anomaly")
ax1.set_ylabel("Log discharge anomaly", color="#1f5fa8")
ax1.tick_params(axis="y", colors="#1f5fa8")
ax1.axhline(0, color="gray", lw=0.6, ls=":")
ax2 = ax1.twinx()
ax2.plot(ym, df["spi_12"], color="#d97706", lw=1.1, alpha=0.85, label="SPI-12")
ax2.set_ylabel("SPI-12 (standardized)", color="#d97706")
ax2.tick_params(axis="y", colors="#d97706")
ax2.spines["right"].set_visible(True)
ax1.set_xlabel("Year")
ax1.set_title("Río Loíza (Caguas gauge) discharge anomaly vs. 12-month SPI, 1959–2026")
fig.tight_layout()
fig.savefig(FIG / "01_discharge_spi_timeseries.png")
plt.close(fig)

# 2. Cross-correlation: discharge anomaly vs SPI-3 at lags -12..+12 months
lags = range(-12, 13)
ccf = []
for lag in lags:
    shifted = df["spi_3"].shift(lag)
    valid = df["log_q_anomaly"].notna() & shifted.notna()
    ccf.append(df.loc[valid, "log_q_anomaly"].corr(shifted[valid]))
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#1f5fa8" if l == 0 else "#4d8fc4" for l in lags]
ax.bar(list(lags), ccf, color=colors, width=0.8)
ax.axvline(0, color="gray", lw=0.6, ls=":")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Lag (months); positive = SPI leads discharge")
ax.set_ylabel("Pearson correlation coefficient")
ax.set_title("Cross-correlation of SPI-3 with discharge anomaly")
fig.tight_layout()
fig.savefig(FIG / "02_crosscorrelation_spi3_discharge.png")
plt.close(fig)

# 3. Observed vs predicted (test set) - best tree model vs persistence
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, test["log_q_anomaly_lag1"], alpha=0.55, s=36, label="Persistence baseline",
           color="#9ca3af", edgecolors="none")
ax.scatter(y_test, en.predict(scaler.transform(X_test)), alpha=0.7, s=36, label="ElasticNet (best model)",
           color="#1f5fa8", edgecolors="none")
lims = [min(y_test.min(), -2), max(y_test.max(), 2)]
ax.plot(lims, lims, color="black", ls="--", lw=1.2, label="1:1 line")
ax.set_xlabel("Observed log discharge anomaly")
ax.set_ylabel("Predicted log discharge anomaly")
ax.set_title("Held-out test set (2011–2026): observed vs. predicted")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "03_obs_vs_pred_test.png")
plt.close(fig)

# 4. SHAP summary (RandomForest, test set)
FEATURE_LABELS = {
    "spi_1": "SPI-1 (1-mo drought index)",
    "spi_3": "SPI-3 (3-mo drought index)",
    "spi_6": "SPI-6 (6-mo drought index)",
    "spi_12": "SPI-12 (12-mo drought index)",
    "precip_mm_sum": "Monthly precipitation (mm)",
    "month_sin": "Seasonal phase (sin)",
    "month_cos": "Seasonal phase (cos)",
    "log_q_anomaly_lag1": "Discharge anomaly, t-1 (persistence)",
}
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
fig = plt.figure(figsize=(8, 5.5))
shap.summary_plot(shap_values, X_test, feature_names=[FEATURE_LABELS[f] for f in FEATURES],
                   plot_type="bar", show=False, color="#1f5fa8")
plt.title("SHAP feature importance — driver attribution\n(RandomForest, held-out test set)")
plt.xlabel("Mean |SHAP value| (impact on predicted log discharge anomaly)")
plt.tight_layout()
plt.savefig(FIG / "04_shap_importance.png")
plt.close(fig)

print(f"\nSaved 4 figures to {FIG}")
