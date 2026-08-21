"""Scatter plots: Stage B NSE vs. each basin characteristic, showing the
actual pattern (not just a summary r) since n=7 is small enough that a
single correlation number can hide real nuance (e.g. Culebrinas: short
shelf distance but a null result -- an outlier worth naming, not hiding)."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
PROC_W = ROOT / "data" / "processed" / "watersheds"
FIG = ROOT / "figures"

df = pd.read_csv(PROC_W / "basin_characteristics_vs_stage_b.csv").dropna(subset=["stage_b_NSE"])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
specs = [
    ("drainage_area_mi2", "Drainage area (mi²)", "r = -0.03 (no relationship)"),
    ("mean_discharge_cfs", "Mean discharge (cfs)", "r = -0.04 (no relationship)"),
    ("shelf_dist_km_to_neg20m", "Distance to -20m isobath (km)", "r = -0.50 (n=7, p=0.25, not significant)"),
]
colors = {"Promising": "#0f766e", "Weak/mixed": "#d97706", "Null": "#c1121f"}
for ax, (col, xlabel, subtitle) in zip(axes, specs):
    for _, row in df.iterrows():
        ax.scatter(row[col], row["stage_b_NSE"], color=colors[row["verdict"]], s=110, zorder=3,
                   edgecolors="white", linewidths=1.0)
        ax.annotate(row["basin"].capitalize(), (row[col], row["stage_b_NSE"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8.5)
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Stage B final-holdout NSE")
    ax.set_title(subtitle, fontsize=10)

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=k)
           for k, c in colors.items()]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=10)
fig.suptitle("What distinguishes basins with a Stage B signal from those without?", fontsize=13, fontweight="bold", y=1.15)
fig.tight_layout()
fig.savefig(FIG / "17_basin_characteristics_vs_stage_b.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG / '17_basin_characteristics_vs_stage_b.png'}")
