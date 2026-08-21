"""
Preprocessing for Stage A (Climate -> Discharge):
- aggregate daily discharge and precip to monthly
- compute log-discharge anomaly (deseasonalized)
- compute SPI-1/3/6/12 via gamma-distribution fit (standard McKee et al. 1993 method)
Outputs data/processed/stage_a_monthly.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# --- Discharge: daily -> monthly mean -> log -> deseasonalized anomaly ---
q = pd.read_csv(RAW / "loiza_caguas_discharge_daily.csv", parse_dates=["date"])
q = q[q["discharge_cfs"] > 0]
q["ym"] = q["date"].dt.to_period("M")
q_monthly = q.groupby("ym")["discharge_cfs"].mean().rename("discharge_cfs_mean")
q_monthly = q_monthly.to_frame()
# CORRECTNESS FIX (found while running the multi-watershed expansion, see
# docs/watershed_candidates.md): reindex to a continuous monthly calendar
# before any rolling/lag computation. Without this, a calendar gap is an
# ABSENT row rather than a NaN row, and .rolling()/.shift() operate on row
# position, not calendar time -- silently splicing non-adjacent months
# together across a gap instead of correctly propagating NaN through it.
q_full_months = pd.period_range(q_monthly.index.min(), q_monthly.index.max(), freq="M")
n_q_gaps = len(q_full_months.difference(q_monthly.index))
print(f"Discharge record has {n_q_gaps} calendar-gap month(s) -- reindexing to expose them as NaN")
q_monthly = q_monthly.reindex(q_full_months)
q_monthly["log_q"] = np.log(q_monthly["discharge_cfs_mean"])
clim = q_monthly.groupby(q_monthly.index.month)["log_q"].transform("mean")
q_monthly["log_q_anomaly"] = q_monthly["log_q"] - clim

# --- Precip: daily -> monthly sum ---
p = pd.read_csv(RAW / "trujillo_alto_precip_daily.csv", parse_dates=["date"])
p["ym"] = p["date"].dt.to_period("M")
p_monthly = p.groupby("ym")["precip_mm"].sum().rename("precip_mm_sum").to_frame()

# Flag months with poor daily coverage (fewer than 20 days reported) as unreliable sums
coverage = p.groupby("ym")["precip_mm"].count().rename("n_days_reported")
p_monthly = p_monthly.join(coverage)
p_monthly.loc[p_monthly["n_days_reported"] < 20, "precip_mm_sum"] = np.nan
p_full_months = pd.period_range(p_monthly.index.min(), p_monthly.index.max(), freq="M")
n_p_gaps = len(p_full_months.difference(p_monthly.index))
print(f"Precip record has {n_p_gaps} calendar-gap month(s) -- reindexing to expose them as NaN")
p_monthly = p_monthly.reindex(p_full_months)


def spi(series: pd.Series, window: int) -> pd.Series:
    """Standardized Precipitation Index via gamma fit (McKee et al. 1993)."""
    accum = series.rolling(window, min_periods=window).sum()
    out = pd.Series(index=series.index, dtype=float)
    for month in range(1, 13):
        mask = accum.index.month == month
        vals = accum[mask].dropna()
        if len(vals) < 10:
            continue
        zeros = (vals == 0).sum()
        p_zero = zeros / len(vals)
        nonzero = vals[vals > 0]
        if len(nonzero) < 5:
            continue
        shape, loc, scale = stats.gamma.fit(nonzero, floc=0)
        cdf = np.where(
            vals == 0,
            p_zero,
            p_zero + (1 - p_zero) * stats.gamma.cdf(vals, shape, loc=loc, scale=scale),
        )
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        out.loc[vals.index] = stats.norm.ppf(cdf)
    return out


p_monthly = p_monthly.sort_index()
for w in (1, 3, 6, 12):
    p_monthly[f"spi_{w}"] = spi(p_monthly["precip_mm_sum"], w)

# --- Month-of-year seasonal encoding ---
month_num = p_monthly.index.month
p_monthly["month_sin"] = np.sin(2 * np.pi * month_num / 12)
p_monthly["month_cos"] = np.cos(2 * np.pi * month_num / 12)

# --- Join ---
stage_a = q_monthly.join(p_monthly, how="inner")
stage_a["log_q_anomaly_lag1"] = stage_a["log_q_anomaly"].shift(1)  # persistence feature
stage_a = stage_a.reset_index().rename(columns={"index": "ym"})
stage_a.to_csv(PROC / "stage_a_monthly.csv", index=False)

print(f"Stage A monthly dataset: {len(stage_a)} rows, {stage_a['ym'].min()} to {stage_a['ym'].max()}")
print(f"Non-null SPI-12 rows: {stage_a['spi_12'].notna().sum()}")
print(stage_a[["ym", "discharge_cfs_mean", "log_q_anomaly", "precip_mm_sum", "spi_3", "spi_12"]].tail(8).to_string(index=False))
