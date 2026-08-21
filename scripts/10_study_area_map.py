"""
Study area map (Figure 1), v4: land-only DEM hillshade clipped to the
accurate Puerto Rico boundary, flat ocean tint (no bathymetry) -- matches
standard hydrology/watershed-paper convention of clipping the DEM to the
study boundary rather than showing full topobathymetry (confirmed via
literature/practice check: DEMs are clipped to the watershed/land boundary
before hillshading in standard workflows, e.g. USU/Purdue hydrology GIS
tutorials, IWA Hydrology Research). Real Rio Grande de Loiza flowline
(USGS NHD) overlaid, with gauge, precip station, and coastal chlorophyll
box marked.

Data sources (all public -- see docs/data_sources.md):
- Elevation: NOAA ERDDAP GEBCO_2020 (15 arc-sec, ~450m/pixel)
- Coastline / clip boundary: UN OCHA COD-AB Puerto Rico admin0 boundary,
  2019-11-08 (user-supplied shapefile, data/pri_adm_2019_shp.zip). This
  replaces two earlier, worse sources tried in this map's history:
    v1 (Census TIGERweb STATE=72, dissolved): had a Census dissolve artifact
       merging Vieques/Culebra into the mainland via a degenerate corridor.
    v2 (Census TIGERweb municipio-level, unary_union'd): fixed the artifact
       but the union only had ~1400 total vertices -- visibly under-detailed
       next to Google Maps when zoomed in.
  The OCHA shapefile has 59 cleanly separate parts (no corridor artifacts);
  the mainland alone is traced at 17,243 vertices -- ~12x the prior boundary's
  detail, extracted once via scripts/ (see pr_mainland_ocha_highres.json) and
  used directly here, no dissolve/union needed.
- River flowlines: USGS National Map Hydro MapServer, layer 5
  (Flowline - Small Scale, HI/PR/VI/Pacific Territories), GNIS_Name LIKE Loiza
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, ListedColormap
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIG = ROOT / "figures"

# Mainland's true bounds (verified via shapely .bounds on the unioned polygon):
# lon [-67.323, -65.423], lat [17.832, 18.568]. Padded here -- the prior fixed
# window (-67.3,-65.55)x(17.9,18.65) clipped ~13km off the real east coast and
# ~7.5km off the real south coast.
XLIM, YLIM = (-67.35, -65.35), (17.78, 18.62)

# ---- Elevation grid (GEBCO_2020, 15 arc-sec, ~450m/pixel) ----
elev = pd.read_csv(RAW / "gebco_2020_pr.csv", skiprows=[1])
elev_grid = elev.pivot(index="latitude", columns="longitude", values="elevation")
lats = elev_grid.index.values
lons = elev_grid.columns.values
Z = elev_grid.values

ls = LightSource(azdeg=315, altdeg=45)
dx_m = np.deg2rad(np.mean(np.diff(lons))) * 6371000 * np.cos(np.deg2rad(np.mean(lats)))
dy_m = np.deg2rad(np.mean(np.diff(lats))) * 6371000
land_norm = Normalize(vmin=0, vmax=float(max(Z.max(), 1)))
# 'terrain' always starts with a blue segment regardless of the data range it's
# normalized to -- skip that segment so low-lying coastal land (elevation ~0m)
# doesn't render as water-blue. Land-only: green (lowland) -> tan -> brown -> white (peaks).
land_cmap = ListedColormap(plt.cm.terrain(np.linspace(0.28, 1.0, 256)))

# ---- PR mainland boundary: OCHA COD-AB, 17,243-vertex mainland ring (see
# module docstring). Used both as the drawn coastline AND as the clip path
# for the DEM, per standard practice of clipping a DEM to the study boundary
# rather than showing full topobathymetry. ----
with open(RAW / "pr_mainland_ocha_highres.json") as f:
    mainland_rings = [json.load(f)["mainland_ring"]]

with open(RAW / "loiza_river_flowlines.geojson") as f:
    river = json.load(f)

sites = {
    "Caguas discharge gauge (USGS 50055000)": (-66.00958889, 18.24269444, "#c1121f", "o"),
    "Trujillo Alto precip. station (GHCN)": (-66.0164, 18.3283, "#fb8500", "^"),
}
# 'far_offshore' box, adopted after testing 4 candidates against the true
# river mouth (-65.878, 18.438, from the NHD flowline endpoint). This box
# showed the strongest, most consistent discharge<->chlorophyll linkage
# (NSE_ElasticNet positive across both folding schemes) -- see
# docs/results_summary.md "Stage B revisited" for the full validation.
COASTAL_BOX = dict(lon_min=-65.93, lon_max=-65.83, lat_min=18.46, lat_max=18.54)
RIVER_MOUTH = (-65.878, 18.438)

fig, ax = plt.subplots(figsize=(9, 5.6))

# Plain white background outside the land boundary (no bathymetry -- not
# relevant to a discharge/precip study; flat blue tint dropped per feedback)
ax.set_facecolor("#ffffff")

# Land-only hillshade
rgb = ls.shade(Z, cmap=land_cmap, norm=land_norm, blend_mode="soft",
                vert_exag=3, dx=dx_m, dy=dy_m)
im = ax.imshow(rgb, extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                origin="lower", zorder=1)

# Clip the raster to the real (clean, municipio-built) PR mainland polygon set
verts, codes = [], []
for ring in mainland_rings:
    verts += ring
    codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY]
clip_path = PathPatch(MplPath(verts, codes), transform=ax.transData, facecolor="none", edgecolor="none")
ax.add_patch(clip_path)
im.set_clip_path(clip_path)

# Coastline outline on top -- dissolved (unary_union) so internal municipio
# borders don't show, only the true outer coastline.
for ring in mainland_rings:
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    ax.plot(xs, ys, color="#1a1a1a", linewidth=1.1, zorder=3)

sm = plt.cm.ScalarMappable(cmap=land_cmap, norm=land_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02, aspect=25)
cbar.set_label("Land elevation (m)", fontsize=10)

# Real river network
for feat in river["features"]:
    geom = feat["geometry"]
    lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
    for line in lines:
        xs, ys = zip(*line)
        ax.plot(xs, ys, color="#1d4ed8", lw=1.8, zorder=4, solid_capstyle="round")

box = Rectangle((COASTAL_BOX["lon_min"], COASTAL_BOX["lat_min"]),
                 COASTAL_BOX["lon_max"] - COASTAL_BOX["lon_min"],
                 COASTAL_BOX["lat_max"] - COASTAL_BOX["lat_min"],
                 facecolor="#0f766e", alpha=0.30, edgecolor="#0f766e", linewidth=1.6, zorder=3,
                 label="Coastal chlorophyll-a box (far_offshore)")
ax.add_patch(box)
ax.scatter(*RIVER_MOUTH, color="#0f766e", marker="D", s=70, zorder=6,
           edgecolors="white", linewidths=1.0, label="Río Loíza mouth (NHD flowline endpoint)")

for label, (lon, lat, color, marker) in sites.items():
    ax.scatter([lon], [lat], color=color, marker=marker, s=150, zorder=6,
               edgecolors="white", linewidths=1.3, label=label)

ax.annotate("Río Grande de Loíza\n(USGS NHD flowline)", xy=(-66.0, 18.33), xytext=(-66.85, 18.13),
            fontsize=9.5, style="italic", color="#1d4ed8",
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.0, connectionstyle="arc3,rad=0.15"))

# North arrow + scale bar (positioned relative to the new, correctly-sized window)
ax.annotate("N", xy=(-65.48, 18.00), xytext=(-65.48, 17.87),
            fontsize=11, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
bar_lon0, bar_lat = -67.28, 17.83
km_per_deg_lon = 111.32 * np.cos(np.radians(18.2))
bar_len_deg = 20 / km_per_deg_lon
ax.plot([bar_lon0, bar_lon0 + bar_len_deg], [bar_lat, bar_lat], color="black", lw=2.5, zorder=6)
ax.text(bar_lon0 + bar_len_deg / 2, bar_lat + 0.015, "20 km", ha="center", fontsize=8.5)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect(1.0)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Study area: Río Grande de Loíza watershed, Puerto Rico")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9.5,
          frameon=True, framealpha=1.0, facecolor="white", edgecolor="#cccccc")

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "08_study_area_map.png")
plt.close(fig)
print(f"Saved study area map to {FIG / '08_study_area_map.png'}")
