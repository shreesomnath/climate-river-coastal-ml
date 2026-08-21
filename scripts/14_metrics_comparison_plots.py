"""
Visualizes the metrics that so far only existed as printed console output +
CSVs: (1) Stage A model-vs-baseline comparison, (2) Stage B's evolution
across box placement and feature-set iterations (original box -> far_offshore
-> +SST/wind), across both folding schemes.
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

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# ============ 1. Stage A: model vs baseline NSE/RMSE/MAE ============
stage_a = pd.read_csv(PROC / "stage_a_model_scores.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
colors = ["#9ca3af" if "Climatology" in m or "Persistence" in m else "#1f5fa8" for m in stage_a["model"]]
for ax, metric in zip(axes, ["NSE", "RMSE", "MAE"]):
    ax.barh(stage_a["model"], stage_a[metric], color=colors)
    ax.set_xlabel(metric)
    ax.axvline(0, color="black", lw=0.8)
    if metric == "NSE":
        ax.set_title("Skill (higher = better)")
    else:
        ax.set_title("Error (lower = better)")
fig.suptitle("Stage A: Climate → Discharge — model vs. baseline comparison", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "11_stage_a_metrics_comparison.png")
plt.close(fig)
print("Saved 11_stage_a_metrics_comparison.png")

# ============ 2. Stage B evolution: box placement + feature set, both folding schemes ============
# Original box (pre-relocation)
orig = pd.read_csv(PROC / "stage_b_cv_scores.csv")
orig_en = orig[["NSE_ElasticNet"]].mean().iloc[0]
orig_en_std = orig[["NSE_ElasticNet"]].std().iloc[0]

# far_offshore, both schemes, base features
fo = pd.read_csv(PROC / "far_offshore_cv_scores.csv")
fo_exp = fo[fo["scheme"] == "expanding_window"]["NSE_EN"]
fo_blk = fo[fo["scheme"] == "blocked_kfold"]["NSE_EN"]

# far_offshore with SST+wind, both schemes
sw = pd.read_csv(PROC / "far_offshore_sst_wind_cv_scores.csv")
sw_base_exp = sw[(sw["feature_set"] == "base") & (sw["scheme"] == "expanding")]["NSE_EN"]
sw_base_blk = sw[(sw["feature_set"] == "base") & (sw["scheme"] == "blocked")]["NSE_EN"]
sw_full_exp = sw[(sw["feature_set"] == "with_sst_wind") & (sw["scheme"] == "expanding")]["NSE_EN"]
sw_full_blk = sw[(sw["feature_set"] == "with_sst_wind") & (sw["scheme"] == "blocked")]["NSE_EN"]

groups = [
    ("Original box\n(pre-relocation)", orig_en, orig_en_std, "#9ca3af"),
    ("far_offshore box\nexpanding-window CV", fo_exp.mean(), fo_exp.std(), "#0f766e"),
    ("far_offshore box\nblocked K-fold CV", fo_blk.mean(), fo_blk.std(), "#0f766e"),
    ("+ SST/wind\nexpanding-window CV", sw_full_exp.mean(), sw_full_exp.std(), "#7c3aed"),
    ("+ SST/wind\nblocked K-fold CV", sw_full_blk.mean(), sw_full_blk.std(), "#7c3aed"),
]
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = [g[0] for g in groups]
means = [g[1] for g in groups]
stds = [g[2] for g in groups]
colors_b = [g[3] for g in groups]
x = np.arange(len(groups))
ax.bar(x, means, yerr=stds, capsize=5, color=colors_b, width=0.6)
ax.axhline(0, color="black", lw=1.0, label="Climatology baseline (NSE=0)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("NSE (ElasticNet), mean ± SD across folds")
ax.set_title("Stage B evolution: box placement and feature-set iterations")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "12_stage_b_evolution.png")
plt.close(fig)
print("Saved 12_stage_b_evolution.png")

# ============ 3. Final-holdout metric suite, far_offshore, all models ============
final_metrics = pd.DataFrame([
    {"model": "Climatology", "NSE": -0.083, "RMSE": 0.265, "MAE": 0.202},
    {"model": "Persistence", "NSE": -0.379, "RMSE": 0.299, "MAE": 0.231},
    {"model": "ElasticNet", "NSE": 0.029, "RMSE": 0.251, "MAE": 0.194},
    {"model": "RandomForest", "NSE": -0.107, "RMSE": 0.268, "MAE": 0.219},
])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
colors_f = ["#9ca3af", "#9ca3af", "#0f766e", "#0f766e"]
axes[0].barh(final_metrics["model"], final_metrics["NSE"], color=colors_f)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_xlabel("NSE")
axes[0].set_title("Final holdout skill")
axes[1].barh(final_metrics["model"], final_metrics["RMSE"], color=colors_f)
axes[1].set_xlabel("RMSE (log chl-a anomaly)")
axes[1].set_title("Final holdout error")
fig.suptitle("far_offshore box: final holdout (last 46 months)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "13_far_offshore_final_holdout.png")
plt.close(fig)
print("Saved 13_far_offshore_final_holdout.png")
