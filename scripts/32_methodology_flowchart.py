"""Publication methodology flowchart: data -> preprocessing -> two-stage
modeling -> validation -> diagnostics. Built with matplotlib patches for
full control and journal-ready output (no external diagram tool needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

fig, ax = plt.subplots(figsize=(11, 15))
ax.set_xlim(0, 10)
ax.set_ylim(0, 18.5)
ax.axis("off")


def box(x, y, w, h, text, color="#e5e7eb", fontsize=11.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        facecolor=color, edgecolor="#374151", linewidth=1.3, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             zorder=3, linespacing=1.35)
    return dict(top=(x + w / 2, y + h), bottom=(x + w / 2, y), cx=x + w / 2, cy=y + h / 2)


def arrow(xy1, xy2, color="#374151", style="-|>"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=15, color=color,
                         linewidth=1.3, zorder=1, shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def stage_label(y, text, num_color="#1f2937"):
    ax.text(0.15, y, text, ha="left", va="center", fontsize=13.5, fontweight="bold", color=num_color)


# ================= Row 1: Data acquisition =================
stage_label(18.1, "1.  Data Acquisition  (public sources, no auth required)")
b1 = box(0.3, 16.5, 2.9, 1.3, "Discharge\nUSGS NWIS\n(10 basins, up to 66 yr)", "#dbeafe")
b2 = box(3.55, 16.5, 2.9, 1.3, "Precipitation\nNOAA GHCN-D\n(nearest long-record station)", "#dbeafe")
b3 = box(6.8, 16.5, 2.9, 1.3, "Chlorophyll-a\nMODIS (2003–22) +\nVIIRS (2012–present)", "#dbeafe")
b4 = box(0.3, 14.9, 2.9, 1.15, "SST (OISST v2.1)\n+ Wind (NCEP/NCAR)", "#eff6ff")
b5 = box(3.55, 14.9, 2.9, 1.15, "River mouth\nUSGS NHD / OpenStreetMap\n(coastline-distance method)", "#eff6ff")
b6 = box(6.8, 14.9, 2.9, 1.15, "Coastline + DEM\nUN OCHA COD-AB / GEBCO", "#eff6ff")

# ================= Row 2: Preprocessing =================
stage_label(13.75, "2.  Preprocessing")
p1 = box(0.5, 12.2, 9.0, 1.3,
         "Continuous-calendar reindex  →  log-transform + deseasonalize (anomalies)  →  SPI-1/3/6/12 (gamma fit)\n"
         "MODIS↔VIIRS inter-sensor bias correction (log-log regression on overlap; per-basin R² tracked, R²=0.11–0.73)",
         "#fef3c7", fontsize=11.0)

for src in [b1, b2, b3, b4, b5, b6]:
    arrow(src["bottom"], (src["cx"], 13.5))

# ================= Row 3: Two-stage modeling =================
stage_label(11.5, "3.  Two-Stage Modeling")
sA = box(0.4, 9.7, 4.3, 1.6,
         "Stage A: Climate → Discharge\nFeatures: SPI×4, precip, season, discharge(t−1)\nTarget: log discharge anomaly",
         "#dcfce7")
sB = box(5.2, 9.7, 4.4, 1.6,
         "Stage B: Discharge → Coastal Chl-a\nFeatures: discharge(t..t−3), SPI-3, season [+SST/wind]\nTarget: log chl-a anomaly (box placement empirically screened)",
         "#dcfce7", fontsize=10.6)
arrow((2.55, 12.2), (2.55, 11.3))
arrow((7.4, 12.2), (7.4, 11.3))

# ================= Row 4: Models + baselines =================
stage_label(9.35, "4.  Models & Baselines")
m1 = box(0.4, 8.05, 4.3, 1.0, "ElasticNet  +  Random Forest\nvs. Climatology & Persistence baselines", "#f3e8ff", fontsize=10.6)
m2 = box(5.2, 8.05, 4.4, 1.0, "ElasticNet  +  Random Forest\nvs. Climatology & Persistence baselines", "#f3e8ff", fontsize=10.6)
arrow(sA["bottom"], m1["top"])
arrow(sB["bottom"], m2["top"])

# ================= Row 5: Validation =================
stage_label(7.65, "5.  Validation")
v1 = box(0.4, 5.7, 9.2, 1.65,
         "Two independent CV schemes: expanding-window TimeSeriesSplit  +  contiguous blocked K-fold (5 folds each)\n"
         "Block-bootstrap 95% CI (1000× resample)   •   Permutation significance test (200× circular-shift null)\n"
         "Learning curves (skill vs. training-set size)",
         "#fee2e2", fontsize=10.8)
arrow(m1["bottom"], (2.55, 7.35))
arrow(m2["bottom"], (7.4, 7.35))

# ================= Row 6: Explainability =================
stage_label(5.35, "6.  Explainability")
e1 = box(1.6, 4.0, 6.8, 1.05, "SHAP (Random Forest, held-out data)\nDriver + lag attribution for every confirmed result", "#e0e7ff", fontsize=11.0)
arrow((5.0, 5.7), (5.0, 5.05))

# ================= Row 7: Diagnostics =================
stage_label(3.7, "7.  Cross-Basin Diagnostics")
d1 = box(0.3, 1.6, 3.0, 1.7, "Basin characteristics\nvs. Stage B skill\n(drainage area, discharge,\nshelf steepness)", "#fce7f3", fontsize=10.4)
d2 = box(3.5, 1.6, 3.0, 1.7, "Bias-correction R²\nvs. Stage B skill\n(r=0.66 — strongest\nrelationship found)", "#fce7f3", fontsize=10.4)
d3 = box(6.7, 1.6, 3.0, 1.7, "MODIS-only re-test\n(confirms confound for\n2 of 3 weak-correction\nbasins)", "#fce7f3", fontsize=10.4)
arrow((5.0, 4.0), (1.8, 3.3))
arrow((5.0, 4.0), (5.0, 3.3))
arrow((5.0, 4.0), (8.2, 3.3))
arrow(d2["bottom"], (5.0, 0.9))

final = box(2.0, 0.15, 6.0, 0.75, "Basin-level result: confirmed / weak-mixed / null / untestable", "#fbbf24", fontsize=11.3)

fig.tight_layout()
fig.savefig(FIG / "22_methodology_flowchart.png", dpi=300)
plt.close(fig)
print(f"Saved {FIG / '22_methodology_flowchart.png'}")
