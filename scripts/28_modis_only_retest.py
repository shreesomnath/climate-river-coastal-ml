"""
Direct test of the bias-correction-confound hypothesis from
docs/watershed_candidates.md: re-run Stage B for the 3 weakest-correction
basins (Culebrinas R2=0.139, Guanajibo R2=0.113, Plata R2=0.475 -- included
as a mid-quality comparison point) using MODIS-ONLY data (2003-01 to
2022-03), no VIIRS splice at all. If these basins look meaningfully
different (less negative / positive) without the noisy VIIRS-corrected
tail, that confirms the sensor-splice-artifact hypothesis. If they look
the same, the null result is more likely real (no linkage), not a data
quality artifact.
"""
import subprocess
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
PROC_W = ROOT / "data" / "processed" / "watersheds"

BASINS = {
    "culebrinas": dict(box_center=(-67.177, 18.406), half=0.05),
    "guanajibo":  dict(box_center=(-67.181, 18.168), half=0.05),
    "plata":      dict(box_center=(-66.196, 18.544), half=0.05),
}
FEATURES = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
            "spi_3", "month_sin", "month_cos"]
TARGET = "log_chl_anomaly"


def fetch_box(box):
    url = (f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday.csv?"
           f"chlorophyll[(2003-01-01):1:(2022-03-16)][({box['lat_min']}):({box['lat_max']})][({box['lon_min']}):({box['lon_max']})]")
    result = subprocess.run(["curl", "-sgL", url], capture_output=True, text=True, timeout=180)
    reader = csv.reader(result.stdout.splitlines()[2:])
    by_month = defaultdict(list)
    for row in reader:
        if len(row) < 4 or row[-1] == "NaN":
            continue
        by_month[row[0][:7]].append(float(row[-1]))
    return by_month


def nse(o, p):
    return 1 - np.sum((o-p)**2) / np.sum((o-o.mean())**2)


def fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=3, random_state=0).fit(scaler.transform(Xtr), ytr)
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0)
    rf.fit(Xtr, ytr)
    return en.predict(scaler.transform(Xte)), rf.predict(Xte)


comparison_rows = []
for name, cfg in BASINS.items():
    print(f"\n{'='*15} {name} (MODIS-only) {'='*15}")
    lon, lat = cfg["box_center"]
    half = cfg["half"]
    box = dict(lon_min=lon-half, lon_max=lon+half, lat_min=lat-half, lat_max=lat+half)
    modis = fetch_box(box)
    print(f"  MODIS: {len(modis)} months")
    chl = pd.DataFrame([(k, np.mean(v)) for k, v in modis.items()], columns=["ym", "chl"])
    chl["ym"] = pd.PeriodIndex(chl["ym"], freq="M")
    chl["log_chl"] = np.log(chl["chl"])
    clim = chl.groupby(chl["ym"].dt.month)["log_chl"].transform("mean")
    chl["log_chl_anomaly"] = chl["log_chl"] - clim

    stage_a = pd.read_csv(PROC_W / f"{name}_stage_a_monthly.csv")
    stage_a["ym"] = pd.PeriodIndex(stage_a[stage_a.columns[0]], freq="M")
    df = stage_a.merge(chl[["ym", "log_chl_anomaly"]], on="ym", how="inner").sort_values("ym").reset_index(drop=True)
    for lag in (0, 1, 2, 3):
        df[f"log_q_anomaly_lag{lag}"] = df["log_q_anomaly"].shift(lag) if lag > 0 else df["log_q_anomaly"]
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"  usable rows: {len(df)} ({df['ym'].iloc[0] if len(df) else '-'} to {df['ym'].iloc[-1] if len(df) else '-'})")
    if len(df) < 50:
        print("  too few rows -- skipping")
        continue

    X, y = df[FEATURES].values, df[TARGET].values
    fold_rows = []
    for scheme_name, splitter in [("expanding", TimeSeriesSplit(n_splits=5)), ("blocked", KFold(n_splits=5, shuffle=False))]:
        for i, (tr, te) in enumerate(splitter.split(X)):
            en_pred, rf_pred = fit_predict(X[tr], y[tr], X[te])
            fold_rows.append({"scheme": scheme_name, "NSE_clim": nse(y[te], np.zeros_like(y[te])),
                               "NSE_EN": nse(y[te], en_pred), "NSE_RF": nse(y[te], rf_pred)})
    fold_df = pd.DataFrame(fold_rows)
    summary = fold_df.groupby("scheme")[["NSE_clim", "NSE_EN", "NSE_RF"]].mean()
    print(summary)

    split = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    en_pred, rf_pred = fit_predict(Xtr, ytr, Xte)
    rho = spearmanr(yte, en_pred)[0]
    print(f"  Final holdout ({len(yte)} mo): NSE_EN={nse(yte,en_pred):+.3f}, Spearman={rho:+.3f}")

    comparison_rows.append({
        "basin": name, "n_rows_modis_only": len(df),
        "NSE_EN_expanding_modis_only": round(summary.loc["expanding", "NSE_EN"], 3),
        "NSE_EN_blocked_modis_only": round(summary.loc["blocked", "NSE_EN"], 3),
        "NSE_EN_final_holdout_modis_only": round(nse(yte, en_pred), 3),
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(PROC_W / "modis_only_retest_results.csv", index=False)
print("\n\n=== MODIS-only vs merged (original) comparison ===")
original = {"culebrinas": {"expanding": -0.055, "blocked": -0.050, "final": -0.059},
            "guanajibo": {"expanding": -0.149, "blocked": -0.105, "final": -0.174},
            "plata": {"expanding": -0.060, "blocked": 0.000, "final": 0.030}}
for _, row in comp_df.iterrows():
    b = row["basin"]
    print(f"{b}: merged(VIIRS-spliced) exp={original[b]['expanding']:+.3f} blk={original[b]['blocked']:+.3f} final={original[b]['final']:+.3f}"
          f"  |  MODIS-only exp={row['NSE_EN_expanding_modis_only']:+.3f} blk={row['NSE_EN_blocked_modis_only']:+.3f} final={row['NSE_EN_final_holdout_modis_only']:+.3f}")
