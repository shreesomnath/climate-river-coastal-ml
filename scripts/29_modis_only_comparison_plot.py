"""Visualize the MODIS-only vs merged (VIIRS-spliced) comparison for the
3 re-tested basins -- the direct test of the sensor-splice-artifact
hypothesis from the Culebrinas investigation."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

data = {
    "culebrinas": {"merged": -0.059, "modis_only": -0.002},
    "guanajibo":  {"merged": -0.174, "modis_only": -0.196},
    "plata":      {"merged": 0.030, "modis_only": 0.100},
}
basins = list(data.keys())
merged_vals = [data[b]["merged"] for b in basins]
modis_vals = [data[b]["modis_only"] for b in basins]

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(basins))
width = 0.35
ax.bar(x - width/2, merged_vals, width, label="Merged (MODIS+VIIRS spliced)", color="#9ca3af")
ax.bar(x + width/2, modis_vals, width, label="MODIS-only (no splice)", color="#0f766e")
ax.axhline(0, color="black", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([b.capitalize() for b in basins])
ax.set_ylabel("Final-holdout NSE (ElasticNet)")
ax.set_title("Testing the sensor-splice-artifact hypothesis:\ndoes removing the noisy VIIRS tail change the result?")
ax.legend()
for i, b in enumerate(basins):
    delta = modis_vals[i] - merged_vals[i]
    note = "improved" if delta > 0.02 else ("worse" if delta < -0.02 else "~same")
    ax.annotate(note, (x[i], max(merged_vals[i], modis_vals[i]) + 0.015), ha="center", fontsize=9, style="italic")
fig.tight_layout()
fig.savefig(FIG / "19_modis_only_retest.png")
plt.close(fig)
print(f"Saved {FIG / '19_modis_only_retest.png'}")
