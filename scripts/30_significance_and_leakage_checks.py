"""
Two rigor additions requested before manuscript drafting:
1. Permutation significance test: is NSE meaningfully better than chance,
   not just better than the climatology baseline? Circularly shift the
   target series (breaks true alignment with features/lags while
   preserving each series' own autocorrelation structure -- the correct
   null for autocorrelated time series, not a naive random shuffle) many
   times, refit, build a null distribution of NSE, get an empirical p-value.
2. SPI leakage sensitivity check: SPI is conventionally fit on the full
   record (standard hydrological practice, produces the most stable
   distribution parameters) but that means test-period precip values
   contribute to the gamma fit used for training months too. Refit SPI
   using ONLY the training period and see if Stage A skill changes.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC_W = PROC / "watersheds"


def nse(o, p):
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def spi(series, window):
    accum = series.rolling(window, min_periods=window).sum()
    out = pd.Series(index=series.index, dtype=float)
    for month in range(1, 13):
        vals = accum[accum.index.month == month].dropna()
        if len(vals) < 10:
            continue
        p_zero = (vals == 0).sum() / len(vals)
        nz = vals[vals > 0]
        if len(nz) < 5:
            continue
        sh, lo, sc = stats.gamma.fit(nz, floc=0)
        cdf = np.where(vals == 0, p_zero, p_zero + (1 - p_zero) * stats.gamma.cdf(vals, sh, loc=lo, scale=sc))
        out.loc[vals.index] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


# ============ 1. PERMUTATION TEST: Loiza Stage A ============
print("=" * 60)
print("PERMUTATION SIGNIFICANCE TEST -- Loiza Stage A (flagship)")
print("=" * 60)
FEATURES_A = ["spi_1", "spi_3", "spi_6", "spi_12", "precip_mm_sum", "month_sin", "month_cos", "log_q_anomaly_lag1"]
TARGET_A = "log_q_anomaly"
df = pd.read_csv(PROC / "stage_a_monthly.csv")
df["ym"] = pd.PeriodIndex(df["ym"], freq="M")
df_valid = df.dropna(subset=FEATURES_A + [TARGET_A]).reset_index(drop=True)
split = int(len(df_valid) * 0.8)
train, test = df_valid.iloc[:split], df_valid.iloc[split:]
Xtr, ytr = train[FEATURES_A].values, train[TARGET_A].values
Xte, yte_real = test[FEATURES_A].values, test[TARGET_A].values

scaler = StandardScaler().fit(Xtr)
en_real = ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), random_state=0).fit(scaler.transform(Xtr), ytr)
real_nse = nse(yte_real, en_real.predict(scaler.transform(Xte)))
print(f"Observed test-set NSE: {real_nse:.3f}")

rng = np.random.default_rng(0)
n_perm = 200
null_nse = []
y_full = df_valid[TARGET_A].values
n = len(y_full)
for i in range(n_perm):
    shift = rng.integers(1, n - 1)
    y_shifted = np.roll(y_full, shift)
    ytr_p, yte_p = y_shifted[:split], y_shifted[split:]
    # refit with the SAME features (only the target's temporal alignment is broken)
    scaler_p = StandardScaler().fit(Xtr)
    en_p = ElasticNetCV(cv=3, random_state=0).fit(scaler_p.transform(Xtr), ytr_p)
    null_nse.append(nse(yte_p, en_p.predict(scaler_p.transform(Xte))))
null_nse = np.array(null_nse)
p_value = (np.sum(null_nse >= real_nse) + 1) / (n_perm + 1)
print(f"Null distribution (circular-shift, n={n_perm}): mean={null_nse.mean():.3f}, "
      f"95th pct={np.percentile(null_nse, 95):.3f}, max={null_nse.max():.3f}")
print(f"Empirical p-value: {p_value:.4f} "
      f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'} at alpha=0.05)")

pd.DataFrame({"null_nse": null_nse}).to_csv(PROC / "permutation_test_loiza_null_dist.csv", index=False)

# ============ 2. PERMUTATION TEST: Manati Stage B (flagship Stage B result) ============
print("\n" + "=" * 60)
print("PERMUTATION SIGNIFICANCE TEST -- Manati Stage B")
print("=" * 60)
FEATURES_B = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
              "spi_3", "month_sin", "month_cos"]
TARGET_B = "log_chl_anomaly"
dfb = pd.read_csv(PROC_W / "manati_stage_b_monthly.csv")
splitb = int(len(dfb) * 0.8)
trainb, testb = dfb.iloc[:splitb], dfb.iloc[splitb:]
Xtrb, ytrb = trainb[FEATURES_B].values, trainb[TARGET_B].values
Xteb, yteb_real = testb[FEATURES_B].values, testb[TARGET_B].values
scalerb = StandardScaler().fit(Xtrb)
enb_real = ElasticNetCV(cv=3, random_state=0).fit(scalerb.transform(Xtrb), ytrb)
real_nse_b = nse(yteb_real, enb_real.predict(scalerb.transform(Xteb)))
print(f"Observed test-set NSE: {real_nse_b:.3f}")

y_full_b = dfb[TARGET_B].values
nb = len(y_full_b)
null_nse_b = []
for i in range(n_perm):
    shift = rng.integers(1, nb - 1)
    y_shifted = np.roll(y_full_b, shift)
    ytr_p, yte_p = y_shifted[:splitb], y_shifted[splitb:]
    scaler_p = StandardScaler().fit(Xtrb)
    en_p = ElasticNetCV(cv=3, random_state=0).fit(scaler_p.transform(Xtrb), ytr_p)
    null_nse_b.append(nse(yte_p, en_p.predict(scaler_p.transform(Xteb))))
null_nse_b = np.array(null_nse_b)
p_value_b = (np.sum(null_nse_b >= real_nse_b) + 1) / (n_perm + 1)
print(f"Null distribution (n={n_perm}): mean={null_nse_b.mean():.3f}, "
      f"95th pct={np.percentile(null_nse_b, 95):.3f}, max={null_nse_b.max():.3f}")
print(f"Empirical p-value: {p_value_b:.4f} "
      f"({'SIGNIFICANT' if p_value_b < 0.05 else 'not significant'} at alpha=0.05)")

# ============ 3. SPI LEAKAGE SENSITIVITY CHECK -- Loiza ============
print("\n" + "=" * 60)
print("SPI LEAKAGE SENSITIVITY CHECK -- Loiza (train-only SPI fit)")
print("=" * 60)
q = pd.read_csv(RAW / "loiza_caguas_discharge_daily.csv", parse_dates=["date"])
q = q[q["discharge_cfs"] > 0]
q["ym"] = q["date"].dt.to_period("M")
q_monthly = q.groupby("ym")["discharge_cfs"].mean().to_frame("discharge_cfs_mean")
q_full = pd.period_range(q_monthly.index.min(), q_monthly.index.max(), freq="M")
q_monthly = q_monthly.reindex(q_full)
q_monthly["log_q"] = np.log(q_monthly["discharge_cfs_mean"])
clim_q = q_monthly.groupby(q_monthly.index.month)["log_q"].transform("mean")
q_monthly["log_q_anomaly"] = q_monthly["log_q"] - clim_q

p = pd.read_csv(RAW / "trujillo_alto_precip_daily.csv", parse_dates=["date"])
p["ym"] = p["date"].dt.to_period("M")
p_monthly = p.groupby("ym")["precip_mm"].sum().to_frame("precip_mm_sum")
coverage = p.groupby("ym")["precip_mm"].count()
p_monthly.loc[coverage[coverage < 20].index, "precip_mm_sum"] = np.nan
p_full = pd.period_range(p_monthly.index.min(), p_monthly.index.max(), freq="M")
p_monthly = p_monthly.reindex(p_full).sort_index()

# Determine the train-period cutoff date (same 80% split point used in the real analysis)
approx_cutoff = df_valid["ym"].iloc[split]
print(f"Using train-only precip data up to {approx_cutoff} to fit SPI distribution parameters")

precip_train_only = p_monthly["precip_mm_sum"].copy()
precip_train_only.loc[precip_train_only.index > approx_cutoff] = np.nan  # hide test-period precip from the fit

for w in (1, 3, 6, 12):
    # fit gamma params on train-only, but still need to score ALL months (train+test) with those params
    accum_full = p_monthly["precip_mm_sum"].rolling(w, min_periods=w).sum()
    accum_train = precip_train_only.rolling(w, min_periods=w).sum()
    out = pd.Series(index=p_monthly.index, dtype=float)
    for month in range(1, 13):
        train_vals = accum_train[accum_train.index.month == month].dropna()
        if len(train_vals) < 10:
            continue
        p_zero = (train_vals == 0).sum() / len(train_vals)
        nz = train_vals[train_vals > 0]
        if len(nz) < 5:
            continue
        sh, lo, sc = stats.gamma.fit(nz, floc=0)
        all_vals = accum_full[accum_full.index.month == month].dropna()
        cdf = np.where(all_vals == 0, p_zero, p_zero + (1 - p_zero) * stats.gamma.cdf(all_vals, sh, loc=lo, scale=sc))
        out.loc[all_vals.index] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    p_monthly[f"spi_{w}_trainfit"] = out

p_monthly["month_sin"] = np.sin(2 * np.pi * p_monthly.index.month / 12)
p_monthly["month_cos"] = np.cos(2 * np.pi * p_monthly.index.month / 12)
df2 = q_monthly.join(p_monthly, how="inner")
df2["log_q_anomaly_lag1"] = df2["log_q_anomaly"].shift(1)
FEATURES_TF = ["spi_1_trainfit", "spi_3_trainfit", "spi_6_trainfit", "spi_12_trainfit",
               "precip_mm_sum", "month_sin", "month_cos", "log_q_anomaly_lag1"]
df2_valid = df2.dropna(subset=FEATURES_TF + [TARGET_A]).reset_index(drop=True)
split2 = int(len(df2_valid) * 0.8)
train2, test2 = df2_valid.iloc[:split2], df2_valid.iloc[split2:]
scaler2 = StandardScaler().fit(train2[FEATURES_TF])
en2 = ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), random_state=0).fit(scaler2.transform(train2[FEATURES_TF]), train2[TARGET_A])
nse_trainfit = nse(test2[TARGET_A].values, en2.predict(scaler2.transform(test2[FEATURES_TF])))
print(f"NSE with full-record SPI fit (original): {real_nse:.3f}")
print(f"NSE with train-only SPI fit (leakage-free): {nse_trainfit:.3f}")
print(f"Difference: {real_nse - nse_trainfit:+.3f}")
