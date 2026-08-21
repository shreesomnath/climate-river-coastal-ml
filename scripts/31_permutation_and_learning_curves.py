"""
Two robustness visualizations flagged as missing:
1. Permutation-test null distribution plots (Loiza Stage A, Manati Stage B)
   -- standard way to report a permutation significance test visually,
   not just as a p-value in text.
2. Learning curves (skill vs. training-set size) for both flagship results
   -- directly addresses the Limitations point about basin/sample size
   adequacy: shows whether NSE is still climbing with more data (more
   basins/longer records would help) or has plateaued (current data is
   sufficient for the model complexity used).
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
PROC_W = PROC / "watersheds"
FIG = ROOT / "figures"


def nse(o, p):
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


# ============ 1. Permutation test null-distribution plots ============
null_loiza = pd.read_csv(PROC / "permutation_test_loiza_null_dist.csv")["null_nse"].values
real_loiza = 0.390

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.hist(null_loiza, bins=25, color="#9ca3af", edgecolor="white", label="Null distribution\n(200 circular-shift permutations)")
ax.axvline(real_loiza, color="#c1121f", lw=2.5, label=f"Observed NSE = {real_loiza:.3f}\n(p = 0.005)")
ax.set_xlabel("NSE")
ax.set_ylabel("Count")
ax.set_title("Loíza Stage A: permutation significance test")
ax.legend(fontsize=9)

# Manati null dist wasn't saved to CSV in script 30 -- regenerate quickly (same seed, reproducible)
FEATURES_B = ["log_q_anomaly_lag0", "log_q_anomaly_lag1", "log_q_anomaly_lag2", "log_q_anomaly_lag3",
              "spi_3", "month_sin", "month_cos"]
TARGET_B = "log_chl_anomaly"
dfb = pd.read_csv(PROC_W / "manati_stage_b_monthly.csv")
splitb = int(len(dfb) * 0.8)
trainb, testb = dfb.iloc[:splitb], dfb.iloc[splitb:]
Xtrb, ytrb = trainb[FEATURES_B].values, trainb[TARGET_B].values
Xteb = testb[FEATURES_B].values
rng = np.random.default_rng(0)
y_full_b = dfb[TARGET_B].values
nb = len(y_full_b)
null_manati = []
for i in range(200):
    shift = rng.integers(1, nb - 1)
    y_shifted = np.roll(y_full_b, shift)
    ytr_p, yte_p = y_shifted[:splitb], y_shifted[splitb:]
    scaler_p = StandardScaler().fit(Xtrb)
    en_p = ElasticNetCV(cv=3, random_state=0).fit(scaler_p.transform(Xtrb), ytr_p)
    null_manati.append(nse(yte_p, en_p.predict(scaler_p.transform(Xteb))))
null_manati = np.array(null_manati)
real_manati = 0.146

ax = axes[1]
ax.hist(null_manati, bins=25, color="#9ca3af", edgecolor="white", label="Null distribution\n(200 circular-shift permutations)")
ax.axvline(real_manati, color="#0f766e", lw=2.5, label=f"Observed NSE = {real_manati:.3f}\n(p = 0.005)")
ax.set_xlabel("NSE")
ax.set_ylabel("Count")
ax.set_title("Manatí Stage B: permutation significance test")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "20_permutation_tests.png")
plt.close(fig)
print(f"Saved {FIG / '20_permutation_tests.png'}")

# ============ 2. Learning curves ============
FEATURES_A = ["spi_1", "spi_3", "spi_6", "spi_12", "precip_mm_sum", "month_sin", "month_cos", "log_q_anomaly_lag1"]
TARGET_A = "log_q_anomaly"
df = pd.read_csv(PROC / "stage_a_monthly.csv")
df_valid = df.dropna(subset=FEATURES_A + [TARGET_A]).reset_index(drop=True)
split = int(len(df_valid) * 0.8)
train_full, test = df_valid.iloc[:split], df_valid.iloc[split:]
Xte, yte = test[FEATURES_A].values, test[TARGET_A].values

train_fracs = np.linspace(0.15, 1.0, 12)
loiza_curve = []
for frac in train_fracs:
    n_use = max(20, int(len(train_full) * frac))
    sub = train_full.iloc[:n_use]  # chronological prefix -- never include future data
    Xtr, ytr = sub[FEATURES_A].values, sub[TARGET_A].values
    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=min(5, max(2, n_use // 40)), random_state=0).fit(scaler.transform(Xtr), ytr)
    loiza_curve.append(nse(yte, en.predict(scaler.transform(Xte))))

manati_curve = []
manati_ns = []
for frac in train_fracs:
    n_use = max(20, int(len(trainb) * frac))
    sub = trainb.iloc[:n_use]
    Xtr, ytr = sub[FEATURES_B].values, sub[TARGET_B].values
    scaler = StandardScaler().fit(Xtr)
    en = ElasticNetCV(cv=min(3, max(2, n_use // 30)), random_state=0).fit(scaler.transform(Xtr), ytr)
    manati_curve.append(nse(testb[TARGET_B].values, en.predict(scaler.transform(Xteb))))
    manati_ns.append(n_use)

loiza_ns = [max(20, int(len(train_full) * f)) for f in train_fracs]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(loiza_ns, loiza_curve, marker="o", color="#c1121f", lw=1.8)
axes[0].axhline(real_loiza, color="gray", lw=1, ls="--", label="Full-data NSE")
axes[0].set_xlabel("Training-set size (months)")
axes[0].set_ylabel("Held-out NSE")
axes[0].set_title("Loíza Stage A: learning curve")
axes[0].legend(fontsize=9)

axes[1].plot(manati_ns, manati_curve, marker="o", color="#0f766e", lw=1.8)
axes[1].axhline(real_manati, color="gray", lw=1, ls="--", label="Full-data NSE")
axes[1].set_xlabel("Training-set size (months)")
axes[1].set_ylabel("Held-out NSE")
axes[1].set_title("Manatí Stage B: learning curve")
axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "21_learning_curves.png")
plt.close(fig)
print(f"Saved {FIG / '21_learning_curves.png'}")

pd.DataFrame({"train_n": loiza_ns, "nse": loiza_curve}).to_csv(PROC / "learning_curve_loiza.csv", index=False)
pd.DataFrame({"train_n": manati_ns, "nse": manati_curve}).to_csv(PROC_W / "learning_curve_manati.csv", index=False)
print("Loiza learning curve:", [round(v, 3) for v in loiza_curve])
print("Manati learning curve:", [round(v, 3) for v in manati_curve])
