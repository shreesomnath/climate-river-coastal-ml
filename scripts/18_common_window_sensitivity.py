"""
Common-window sensitivity check: re-run Stage A for all 8 basins (7 new +
Loiza) restricted to the SAME calendar window, to check the multi-basin
comparison isn't an artifact of different basins using different historical
eras. Each basin's own full-record numbers remain the primary result (see
watershed_candidates.md); this is the robustness check owed on top of that.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
PROC_W = PROC / "watersheds"

FEATURES = ["spi_1", "spi_3", "spi_6", "spi_12", "precip_mm_sum", "month_sin", "month_cos", "log_q_anomaly_lag1"]
TARGET = "log_q_anomaly"

DATASETS = {
    "loiza": PROC / "stage_a_monthly.csv",
    "manati": PROC_W / "manati_stage_a_monthly.csv",
    "plata": PROC_W / "plata_stage_a_monthly.csv",
    "fajardo": PROC_W / "fajardo_stage_a_monthly.csv",
    "anasco": PROC_W / "anasco_stage_a_monthly.csv",
    "patillas": PROC_W / "patillas_stage_a_monthly.csv",
    "culebrinas": PROC_W / "culebrinas_stage_a_monthly.csv",
    "guanajibo": PROC_W / "guanajibo_stage_a_monthly.csv",
}

data = {}
for name, path in DATASETS.items():
    df = pd.read_csv(path)
    idx_col = df.columns[0]
    df["ym"] = pd.PeriodIndex(df[idx_col], freq="M")
    df = df.dropna(subset=FEATURES + [TARGET])
    data[name] = df
    print(f"{name}: {df['ym'].min()} to {df['ym'].max()}, n={len(df)}")

starts = {n: d["ym"].min() for n, d in data.items()}
ends = {n: d["ym"].max() for n, d in data.items()}
common_start_all8 = max(starts.values())
common_end_all8 = min(ends.values())
print(f"\nCommon window across ALL 8: {common_start_all8} to {common_end_all8} "
      f"(bottleneck: start={max(starts, key=starts.get)}, end={min(ends, key=ends.get)})")

# Plata's short precip record dominates the all-8 window -- also compute a
# 7-basin (excluding Plata) common window, which is far more informative
starts_no_plata = {n: s for n, s in starts.items() if n != "plata"}
ends_no_plata = {n: e for n, e in ends.items() if n != "plata"}
common_start_7 = max(starts_no_plata.values())
common_end_7 = min(ends_no_plata.values())
print(f"Common window excluding Plata (7 basins): {common_start_7} to {common_end_7} "
      f"(bottleneck: start={max(starts_no_plata, key=starts_no_plata.get)}, "
      f"end={min(ends_no_plata, key=ends_no_plata.get)})")


def nse(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)


def run_window(win_start, win_end, label, include_plata=True):
    print(f"\n=== {label}: {win_start} to {win_end} ===")
    rows = []
    for name, df in data.items():
        if not include_plata and name == "plata":
            continue
        sub = df[(df["ym"] >= win_start) & (df["ym"] <= win_end)]
        if len(sub) < 40:
            print(f"  {name}: only {len(sub)} months in window -- skipping")
            rows.append({"watershed": name, "n": len(sub), "status": "TOO_SHORT"})
            continue
        split = int(len(sub) * 0.8)
        train, test = sub.iloc[:split], sub.iloc[split:]
        Xtr, ytr = train[FEATURES].values, train[TARGET].values
        Xte, yte = test[FEATURES].values, test[TARGET].values
        scaler = StandardScaler().fit(Xtr)
        en = ElasticNetCV(cv=min(5, max(2, len(train)//30)), random_state=0).fit(scaler.transform(Xtr), ytr)
        en_pred = en.predict(scaler.transform(Xte))
        rf = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=4, random_state=0)
        rf.fit(Xtr, ytr)
        rf_pred = rf.predict(Xte)
        row = {
            "watershed": name, "n": len(sub), "n_test": len(test), "status": "OK",
            "NSE_climatology": round(nse(yte, np.zeros_like(yte)), 3),
            "NSE_ElasticNet": round(nse(yte, en_pred), 3),
            "NSE_RandomForest": round(nse(yte, rf_pred), 3),
        }
        print(f"  {name}: n={len(sub)} NSE_clim={row['NSE_climatology']:+.3f} "
              f"NSE_EN={row['NSE_ElasticNet']:+.3f} NSE_RF={row['NSE_RandomForest']:+.3f}")
        rows.append(row)
    return pd.DataFrame(rows)


res_7 = run_window(common_start_7, common_end_7, "Common window, 7 basins (excl. Plata)", include_plata=False)
res_7.to_csv(PROC_W / "common_window_7basins_results.csv", index=False)

print("\n\n=== SUMMARY: common-window vs full-record NSE_ElasticNet ===")
full_record = pd.read_csv(PROC_W / "stage_a_all_watersheds_comparison.csv")
loiza_full = {"watershed": "loiza", "NSE_ElasticNet": 0.390}  # from results_summary.md, corrected run
full_record = pd.concat([full_record, pd.DataFrame([loiza_full])], ignore_index=True)
compare = res_7.merge(full_record[["watershed", "NSE_ElasticNet"]], on="watershed",
                       suffixes=("_common_window", "_full_record"))
print(compare[["watershed", "NSE_ElasticNet_full_record", "NSE_ElasticNet_common_window"]].to_string(index=False))
