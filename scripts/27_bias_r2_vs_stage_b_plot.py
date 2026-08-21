"""
Refined 'why does the signal appear in some basins' analysis: MODIS-VIIRS
bias-correction R^2 (a data-quality measure, not a physical basin property)
correlates more strongly with Stage B skill than any physical basin
characteristic tested in script 23/24. Mechanistic story: a poor bias
correction means a noisier merged chlorophyll series, which attenuates
any true discharge-chlorophyll correlation regardless of whether the real
physical linkage is present or absent -- this is a methodological confound
to disclose, not evidence the linkage itself is basin-dependent.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
PROC_W = ROOT / "data" / "processed" / "watersheds"

data = {
    "Loíza": (0.688, 0.029, "Promising"), "Manatí": (0.730, 0.146, "Promising"),
    "Plata": (0.475, 0.030, "Null"), "Añasco": (0.475, 0.063, "Weak/mixed"),
    "Culebrinas": (0.139, -0.059, "Null"), "Patillas": (0.182, 0.096, "Weak/mixed"),
    "Guanajibo": (0.113, -0.174, "Null"),
}
df = pd.DataFrame(data, index=["bias_r2", "stage_b_nse", "verdict"]).T
df["bias_r2"] = df["bias_r2"].astype(float)
df["stage_b_nse"] = df["stage_b_nse"].astype(float)
df.to_csv(PROC_W / "bias_r2_vs_stage_b.csv")

r, p = stats.pearsonr(df["bias_r2"], df["stage_b_nse"])

colors = {"Promising": "#0f766e", "Weak/mixed": "#d97706", "Null": "#c1121f"}
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, row in df.iterrows():
    ax.scatter(row["bias_r2"], row["stage_b_nse"], color=colors[row["verdict"]], s=130, zorder=3,
               edgecolors="white", linewidths=1.2)
    ax.annotate(name, (row["bias_r2"], row["stage_b_nse"]), xytext=(6, 6), textcoords="offset points", fontsize=9.5)
ax.axhline(0, color="gray", lw=0.7, ls=":")
ax.set_xlabel("MODIS↔VIIRS bias-correction R² (data-quality proxy)")
ax.set_ylabel("Stage B final-holdout NSE")
ax.set_title(f"Bias-correction quality vs. Stage B skill\nr={r:.3f}, p={p:.3f} (n=7, suggestive not significant)",
             fontsize=12)
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=k)
           for k, c in colors.items()]
ax.legend(handles=handles, loc="lower right", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG / "18_bias_r2_vs_stage_b.png")
plt.close(fig)
print(f"Saved {FIG / '18_bias_r2_vs_stage_b.png'}")
print(f"r={r:.3f}, p={p:.3f}")
