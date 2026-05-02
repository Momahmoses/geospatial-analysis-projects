"""
Flood Risk & Early Warning System
Combines elevation, rainfall intensity, river proximity, and soil drainage
to produce multi-factor flood risk maps and trigger zone-level alerts.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon, LineString
import folium
from folium.plugins import HeatMap
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(7)

CITY_LAT, CITY_LON = 4.0435, 9.7043   # Douala, Cameroon — flood-prone city
GRID_SIZE = 60
SPACING_M = 250

lat_step = SPACING_M / 111320
lon_step = SPACING_M / (111320 * np.cos(np.radians(CITY_LAT)))

# ---------------------------------------------------------------------------
# 1. Build terrain grid with realistic elevation model
# ---------------------------------------------------------------------------
def perlin_like_noise(rows, cols, scale=10):
    """Simple smooth noise approximating terrain."""
    from scipy.ndimage import gaussian_filter
    raw = np.random.rand(rows, cols)
    return gaussian_filter(raw, sigma=scale)

try:
    from scipy.ndimage import gaussian_filter
    elev_noise = perlin_like_noise(GRID_SIZE, GRID_SIZE, scale=8)
except ImportError:
    elev_noise = np.random.rand(GRID_SIZE, GRID_SIZE)

# Elevation: river valley runs diagonally, flanked by hills
elev_base = np.zeros((GRID_SIZE, GRID_SIZE))
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        dist_valley = abs(r - c) / GRID_SIZE
        elev_base[r, c] = 5 + 60 * dist_valley**1.5 + 30 * elev_noise[r, c]

records = []
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        lat = CITY_LAT + r * lat_step
        lon = CITY_LON + c * lon_step
        poly = Polygon([
            (lon, lat), (lon + lon_step, lat),
            (lon + lon_step, lat + lat_step), (lon, lat + lat_step),
        ])
        records.append({
            "row": r, "col": c, "lat": lat + lat_step/2, "lon": lon + lon_step/2,
            "elevation": elev_base[r, c], "geometry": poly
        })

grid = gpd.GeoDataFrame(records, crs="EPSG:4326")

# ---------------------------------------------------------------------------
# 2. River network — main river + two tributaries
# ---------------------------------------------------------------------------
river_cells = set()
for r in range(GRID_SIZE):
    c = r  # diagonal main river
    if 0 <= c < GRID_SIZE:
        river_cells.add((r, c))
        if c+1 < GRID_SIZE:
            river_cells.add((r, c+1))
# Tributary 1
for r in range(GRID_SIZE//2):
    c = GRID_SIZE//4 + r//2
    if 0 <= c < GRID_SIZE:
        river_cells.add((r, c))

grid["near_river"] = grid.apply(
    lambda x: 1 if (x["row"], x["col"]) in river_cells else 0, axis=1
)

# Distance to nearest river cell (in grid units)
def river_distance(row, col):
    return min(np.sqrt((row - rr)**2 + (col - cc)**2) for rr, cc in river_cells)

sample_dist = grid.apply(lambda x: river_distance(x["row"], x["col"]), axis=1)
grid["river_dist"] = sample_dist

# ---------------------------------------------------------------------------
# 3. Rainfall intensity (mm/hr) — seasonal storm simulation
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import gaussian_filter
    rain_raw = np.random.exponential(30, (GRID_SIZE, GRID_SIZE))
    rain_smooth = gaussian_filter(rain_raw, sigma=5)
except ImportError:
    rain_smooth = np.random.exponential(30, (GRID_SIZE, GRID_SIZE))

grid["rainfall_mm"] = [rain_smooth[r, c] for r, c in zip(grid["row"], grid["col"])]
grid["rainfall_mm"] = grid["rainfall_mm"].clip(5, 150)

# ---------------------------------------------------------------------------
# 4. Soil drainage capacity (0=poor, 1=excellent)
# ---------------------------------------------------------------------------
DRAIN_NOISE = np.random.beta(2, 3, (GRID_SIZE, GRID_SIZE))
grid["soil_drainage"] = [DRAIN_NOISE[r, c] for r, c in zip(grid["row"], grid["col"])]
# Urban/impervious areas near city center have poor drainage
center = GRID_SIZE // 2
grid["urban_dist"] = np.sqrt((grid["row"] - center)**2 + (grid["col"] - center)**2)
grid["soil_drainage"] = grid["soil_drainage"] * (grid["urban_dist"] / GRID_SIZE).clip(0.1, 1)

# ---------------------------------------------------------------------------
# 5. Composite flood risk score (0–100)
# ---------------------------------------------------------------------------
# Normalise factors
elev_score = 1 - (grid["elevation"] - grid["elevation"].min()) / (grid["elevation"].max() - grid["elevation"].min())
rain_score = (grid["rainfall_mm"] - grid["rainfall_mm"].min()) / (grid["rainfall_mm"].max() - grid["rainfall_mm"].min())
river_score = 1 - (grid["river_dist"].clip(0, 15) / 15)
drain_score = 1 - grid["soil_drainage"]

# Weighted composite
grid["flood_risk"] = (
    0.35 * elev_score +
    0.25 * rain_score +
    0.25 * river_score +
    0.15 * drain_score
) * 100

def classify_flood(score):
    if score >= 75: return "Extreme"
    elif score >= 55: return "High"
    elif score >= 35: return "Moderate"
    elif score >= 15: return "Low"
    else: return "Minimal"

grid["flood_class"] = grid["flood_risk"].apply(classify_flood)

# ---------------------------------------------------------------------------
# 6. Static 3-panel map
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("#0a0a1a")
for ax in axes:
    ax.set_facecolor("#0a0a1a")

# Elevation
norm_e = mcolors.Normalize(vmin=grid["elevation"].min(), vmax=grid["elevation"].max())
grid.plot(column="elevation", cmap="terrain", norm=norm_e, ax=axes[0], linewidth=0)
sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm_e)
plt.colorbar(sm, ax=axes[0], label="Elevation (m)", shrink=0.8)
axes[0].set_title("Terrain Elevation Model", color="white", fontsize=12, fontweight="bold")
axes[0].tick_params(colors="white")

# Rainfall
norm_r = mcolors.Normalize(0, 150)
grid.plot(column="rainfall_mm", cmap="Blues", norm=norm_r, ax=axes[1], linewidth=0)
sm2 = plt.cm.ScalarMappable(cmap="Blues", norm=norm_r)
plt.colorbar(sm2, ax=axes[1], label="Rainfall (mm/hr)", shrink=0.8)
axes[1].set_title("Rainfall Intensity", color="white", fontsize=12, fontweight="bold")
axes[1].tick_params(colors="white")

# Flood risk
risk_cmap = mcolors.ListedColormap(["#1565c0", "#4caf50", "#fdd835", "#ef6c00", "#b71c1c"])
risk_bounds = [0, 15, 35, 55, 75, 100]
risk_norm = mcolors.BoundaryNorm(risk_bounds, risk_cmap.N)
grid.plot(column="flood_risk", cmap=risk_cmap, norm=risk_norm, ax=axes[2], linewidth=0)
legend_elements = [
    Patch(facecolor="#1565c0", label="Minimal (<15)"),
    Patch(facecolor="#4caf50", label="Low (15–35)"),
    Patch(facecolor="#fdd835", label="Moderate (35–55)"),
    Patch(facecolor="#ef6c00", label="High (55–75)"),
    Patch(facecolor="#b71c1c", label="Extreme (>75)"),
]
axes[2].legend(handles=legend_elements, loc="lower right", framealpha=0.8,
               facecolor="#111", labelcolor="white", fontsize=8)
axes[2].set_title("Composite Flood Risk Score", color="white", fontsize=12, fontweight="bold")
axes[2].tick_params(colors="white")

plt.suptitle("Flood Risk & Early Warning Analysis", color="white",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/flood_risk_map.png", dpi=180, bbox_inches="tight", facecolor="#0a0a1a")
plt.close()
print("Saved: outputs/flood_risk_map.png")

# ---------------------------------------------------------------------------
# 7. Risk distribution chart
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor("#0a0a1a")
ax2.set_facecolor("#111")
order = ["Extreme", "High", "Moderate", "Low", "Minimal"]
colors = ["#b71c1c", "#ef6c00", "#fdd835", "#4caf50", "#1565c0"]
counts = [len(grid[grid["flood_class"] == c]) for c in order]
bars = ax2.bar(order, counts, color=colors, edgecolor="none")
ax2.set_ylabel("Number of Grid Cells", color="white")
ax2.set_title("Flood Risk Zone Distribution", color="white", fontsize=13, fontweight="bold")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("#444")
for bar, val in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha="center", color="white", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/flood_risk_distribution.png", dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
plt.close()
print("Saved: outputs/flood_risk_distribution.png")

# ---------------------------------------------------------------------------
# 8. Interactive early warning map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.04, CITY_LON + 0.04], zoom_start=12,
               tiles="CartoDB dark_matter")

for _, row in grid[grid["flood_class"].isin(["Extreme", "High"])].iterrows():
    color = "#b71c1c" if row["flood_class"] == "Extreme" else "#ef6c00"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=5,
        color=color, fill=True, fill_opacity=0.75,
        popup=folium.Popup(
            f"<b>⚠ {row['flood_class'].upper()} FLOOD RISK</b><br>"
            f"Risk Score: {row['flood_risk']:.1f}/100<br>"
            f"Elevation: {row['elevation']:.1f} m<br>"
            f"Rainfall: {row['rainfall_mm']:.0f} mm/hr<br>"
            f"River Dist: {row['river_dist']:.1f} cells",
            max_width=220
        )
    ).add_to(m)

HeatMap(grid[["lat", "lon", "flood_risk"]].values.tolist(),
        min_opacity=0.3, radius=16, blur=12,
        gradient={"0.2": "#1565c0", "0.5": "#fdd835", "0.8": "#ef6c00", "1.0": "#b71c1c"},
        name="Flood Risk Heatmap").add_to(m)

title_html = """
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.85);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:16px;font-weight:bold;z-index:9999;">
Flood Risk Early Warning System
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("outputs/flood_risk_interactive.html")
print("Saved: outputs/flood_risk_interactive.html")

# ---------------------------------------------------------------------------
# 9. Early warning alerts
# ---------------------------------------------------------------------------
extreme_zones = grid[grid["flood_class"] == "Extreme"]
high_zones = grid[grid["flood_class"] == "High"]

print("\n" + "="*60)
print("  FLOOD RISK EARLY WARNING REPORT")
print("="*60)
print(f"  Total zones analyzed     : {len(grid):,}")
print(f"  EXTREME risk zones       : {len(extreme_zones):,} ({100*len(extreme_zones)/len(grid):.1f}%)")
print(f"  HIGH risk zones          : {len(high_zones):,} ({100*len(high_zones)/len(grid):.1f}%)")
print(f"  Max flood risk score     : {grid['flood_risk'].max():.1f}/100")
print(f"  Mean elevation (extreme) : {extreme_zones['elevation'].mean():.1f} m")
print(f"  Mean rainfall (extreme)  : {extreme_zones['rainfall_mm'].mean():.1f} mm/hr")
print("\n  ACTIVE ALERTS:")
for _, row in extreme_zones.head(5).iterrows():
    print(f"  🚨 Zone ({row['row']},{row['col']}) — Score:{row['flood_risk']:.0f} | "
          f"Elev:{row['elevation']:.0f}m | Rain:{row['rainfall_mm']:.0f}mm/hr")
print("="*60)

grid.drop(columns=["geometry"]).to_csv("outputs/flood_risk_data.csv", index=False)
print("\nSaved: outputs/flood_risk_data.csv")
