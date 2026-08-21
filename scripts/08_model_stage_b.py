"""
Stage B modeling: Discharge (lag 0-3mo) + SPI-3 + seasonality -> coastal
chlorophyll-a anomaly. Same protocol as Stage A (04_model_stage_a.py) but
with expanding-window CV throughout instead of a single holdout, since the
VIIRS-based record is much shorter (n~148 vs n~570 in Stage A).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import shap

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

FEATURES = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2",
            "log_q_anomaly_lag3", "spi_3", "month_sin", "month_cos"]
TARGET = "log_chl_anomaly"

df = pd.read_csv(PROC / "stage_b_monthly.csv")
df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
print(f"Usable rows: {len(df)} ({df['ym'].iloc[0]} to {df['ym'].iloc[-1]})")


def nse(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)


def rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


# ---- Expanding-window CV (5 folds) across the whole record ----
tscv = TimeSeriesSplit(n_splits=5)
X, y = df[FEATURES].values, df[TARGET].values

fold_rows = []
last_fold_test_idx, last_fold_preds = None, {}
for i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
    Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    clim_pred = np.zeros_like(yte)
    persist_pred = df["log_chl_anomaly"].shift(1).values[te_idx]
    valid = ~np.isnan(persist_pred)

    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=3, random_state=0).fit(scaler.transform(Xtr), ytr)
    en_pred = en.predict(scaler.transform(Xte))

    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)

    fold_rows.append({
        "fold": i + 1, "n_test": len(te_idx),
        "NSE_climatology": round(nse(yte, clim_pred), 3),
        "NSE_persistence": round(nse(yte[valid], persist_pred[valid]), 3) if valid.sum() > 2 else np.nan,
        "NSE_ElasticNet": round(nse(yte, en_pred), 3),
        "NSE_RandomForest": round(nse(yte, rf_pred), 3),
        "RMSE_ElasticNet": round(rmse(yte, en_pred), 3),
    })
    if i == tscv.n_splits - 1:
        last_fold_test_idx, last_fold_preds = te_idx, {"obs": yte, "en": en_pred, "rf": rf_pred}
        last_rf_model, last_Xte = rf, Xte

results = pd.DataFrame(fold_rows)
print("\n=== Stage B: expanding-window CV skill (5 folds) ===")
print(results.to_string(index=False))
print("\nMean +/- SD across folds:")
for col in ["NSE_climatology", "NSE_persistence", "NSE_ElasticNet", "NSE_RandomForest"]:
    print(f"  {col}: {results[col].mean():.3f} +/- {results[col].std():.3f}")
results.to_csv(PROC / "stage_b_cv_scores.csv", index=False)

# =================== PLOTS ===================

# 5. River-to-coast lag scan: discharge anomaly vs chlorophyll anomaly
lags = range(0, 7)
ccf = []
for lag in lags:
    shifted = df["log_q_anomaly"].shift(lag)
    valid = df["log_chl_anomaly"].notna() & shifted.notna()
    ccf.append(df.loc[valid, "log_chl_anomaly"].corr(shifted[valid]))
fig, ax = plt.subplots(figsize=(8, 4.5))
best_lag = int(np.nanargmax(np.abs(ccf)))
colors = ["#b45309" if i == best_lag else "#d9a066" for i in range(len(lags))]
ax.bar(list(lags), ccf, color=colors, width=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("River discharge lag (months before coastal observation)")
ax.set_ylabel("Pearson correlation coefficient")
ax.set_title("River-to-coast lag scan: discharge anomaly vs.\ncoastal chlorophyll-a anomaly")
fig.tight_layout()
fig.savefig(FIG / "05_river_to_coast_lag_scan.png")
plt.close(fig)
print(f"\nStrongest river-to-coast correlation at lag = {best_lag} month(s), r = {ccf[best_lag]:.3f}")

# 6. Time series: chlorophyll anomaly vs discharge anomaly (VIIRS period)
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ym = pd.PeriodIndex(df["ym"], freq="M").to_timestamp()
ax1.plot(ym, df["log_chl_anomaly"], color="#0f766e", lw=1.2, label="Coastal chlorophyll-a anomaly")
ax1.set_ylabel("Log chlorophyll-a anomaly", color="#0f766e")
ax1.tick_params(axis="y", colors="#0f766e")
ax1.axhline(0, color="gray", lw=0.6, ls=":")
ax2 = ax1.twinx()
ax2.plot(ym, df["log_q_anomaly"], color="#1f5fa8", lw=1.0, alpha=0.8, label="Discharge anomaly")
ax2.set_ylabel("Log discharge anomaly", color="#1f5fa8")
ax2.tick_params(axis="y", colors="#1f5fa8")
ax1.set_xlabel("Year")
ax1.set_title("Coastal chlorophyll-a anomaly near Loíza mouth vs. upstream\ndischarge anomaly, 2012–2026 (VIIRS)")
fig.tight_layout()
fig.savefig(FIG / "06_chl_discharge_timeseries.png")
plt.close(fig)

# 7. SHAP on the last fold's RF model
FEATURE_LABELS = {
    "log_q_anomaly_lag0": "Discharge anomaly, t (same month)",
    "log_q_anomaly_lag1": "Discharge anomaly, t-1",
    "log_q_anomaly_lag2": "Discharge anomaly, t-2",
    "log_q_anomaly_lag3": "Discharge anomaly, t-3",
    "spi_3": "SPI-3 (3-mo drought index)",
    "month_sin": "Seasonal phase (sin)",
    "month_cos": "Seasonal phase (cos)",
}
explainer = shap.TreeExplainer(last_rf_model)
shap_values = explainer.shap_values(last_Xte)
fig = plt.figure(figsize=(8, 5.5))
shap.summary_plot(shap_values, last_Xte, feature_names=[FEATURE_LABELS[f] for f in FEATURES],
                   plot_type="bar", show=False, color="#0f766e")
plt.title("SHAP feature importance — river-to-coast driver attribution\n(RandomForest, final CV fold)")
plt.xlabel("Mean |SHAP value| (impact on predicted chlorophyll anomaly)")
plt.tight_layout()
plt.savefig(FIG / "07_shap_stage_b.png")
plt.close(fig)

print(f"\nSaved 3 figures to {FIG}")
