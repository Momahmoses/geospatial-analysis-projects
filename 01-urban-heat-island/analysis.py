"""
Urban Heat Island (UHI) Mapping
Identifies city zones with dangerous heat buildup using land surface temperature
simulation and land-use data. Guides tree planting and cool-roof policy decisions.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon
import folium
from folium.plugins import HeatMap
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic city grid — mimics a mid-size African/Asian city (10x10 km)
# ---------------------------------------------------------------------------
CITY_NAME = "MetroCity"
CITY_LAT, CITY_LON = 6.5244, 3.3792   # Lagos-like coordinates

def generate_city_grid(rows=50, cols=50, spacing_m=200):
    """Generate a 50x50 grid of 200m cells representing city zones."""
    lat_step = spacing_m / 111320
    lon_step = spacing_m / (111320 * np.cos(np.radians(CITY_LAT)))

    records = []
    for r in range(rows):
        for c in range(cols):
            lat = CITY_LAT + r * lat_step
            lon = CITY_LON + c * lon_step
            poly = Polygon([
                (lon, lat),
                (lon + lon_step, lat),
                (lon + lon_step, lat + lat_step),
                (lon, lat + lat_step),
            ])
            records.append({"row": r, "col": c, "lat": lat + lat_step/2,
                             "lon": lon + lon_step/2, "geometry": poly})
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

grid = generate_city_grid()

# ---------------------------------------------------------------------------
# 2. Assign land-use types with realistic spatial distribution
# ---------------------------------------------------------------------------
center_r, center_c = 25, 25

def distance_from_center(row, col):
    return np.sqrt((row - center_r)**2 + (col - center_c)**2)

grid["dist_center"] = grid.apply(lambda x: distance_from_center(x["row"], x["col"]), axis=1)

def assign_land_use(row, col, dist):
    if dist < 5:
        return "Commercial/CBD"
    elif dist < 10:
        return "Dense Residential"
    elif dist < 16:
        choices = ["Industrial", "Mixed Use", "Dense Residential"]
        weights = [0.3, 0.3, 0.4]
        return np.random.choice(choices, p=weights)
    elif dist < 22:
        choices = ["Sparse Residential", "Green Space", "Mixed Use"]
        weights = [0.5, 0.3, 0.2]
        return np.random.choice(choices, p=weights)
    else:
        choices = ["Sparse Residential", "Green Space", "Agricultural"]
        weights = [0.3, 0.4, 0.3]
        return np.random.choice(choices, p=weights)

grid["land_use"] = grid.apply(lambda x: assign_land_use(x["row"], x["col"], x["dist_center"]), axis=1)

# ---------------------------------------------------------------------------
# 3. Simulate Land Surface Temperature (LST) in Celsius
#    Based on land use, impervious surface fraction, and vegetation index
# ---------------------------------------------------------------------------
LST_BASE = {
    "Commercial/CBD": 42.0,
    "Dense Residential": 39.5,
    "Industrial": 41.0,
    "Mixed Use": 38.0,
    "Sparse Residential": 35.5,
    "Green Space": 31.0,
    "Agricultural": 30.5,
}

NDVI_BASE = {
    "Commercial/CBD": 0.05,
    "Dense Residential": 0.12,
    "Industrial": 0.08,
    "Mixed Use": 0.18,
    "Sparse Residential": 0.30,
    "Green Space": 0.72,
    "Agricultural": 0.65,
}

grid["LST"] = grid["land_use"].map(LST_BASE) + np.random.normal(0, 1.2, len(grid))
grid["NDVI"] = grid["land_use"].map(NDVI_BASE) + np.random.normal(0, 0.03, len(grid)).clip(-0.05, 0.05)
grid["NDVI"] = grid["NDVI"].clip(0.02, 0.95)

# Impervious surface fraction (0–1)
ISF_BASE = {"Commercial/CBD": 0.92, "Dense Residential": 0.78, "Industrial": 0.85,
            "Mixed Use": 0.60, "Sparse Residential": 0.45, "Green Space": 0.05, "Agricultural": 0.10}
grid["ISF"] = grid["land_use"].map(ISF_BASE) + np.random.normal(0, 0.04, len(grid))
grid["ISF"] = grid["ISF"].clip(0, 1)

# UHI intensity relative to rural fringe (cells with dist > 20)
rural_mean_lst = grid[grid["dist_center"] > 20]["LST"].mean()
grid["UHI_intensity"] = grid["LST"] - rural_mean_lst

# Risk classification
def classify_risk(uhi):
    if uhi >= 6: return "Critical"
    elif uhi >= 4: return "High"
    elif uhi >= 2: return "Moderate"
    elif uhi >= 0: return "Low"
    else: return "Cool Zone"

grid["risk"] = grid["UHI_intensity"].apply(classify_risk)

# ---------------------------------------------------------------------------
# 4. Static map — LST choropleth
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("#0f0f0f")

for ax in axes:
    ax.set_facecolor("#0f0f0f")

# Panel A: Land Surface Temperature
norm = mcolors.Normalize(vmin=grid["LST"].min(), vmax=grid["LST"].max())
cmap = plt.cm.RdYlBu_r
grid.plot(column="LST", cmap=cmap, norm=norm, ax=axes[0], linewidth=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, ax=axes[0], label="LST (°C)", shrink=0.8)
axes[0].set_title("Land Surface Temperature", color="white", fontsize=13, fontweight="bold")
axes[0].tick_params(colors="white")
axes[0].set_xlabel("Longitude", color="white")
axes[0].set_ylabel("Latitude", color="white")

# Panel B: NDVI
norm2 = mcolors.Normalize(vmin=0, vmax=0.8)
grid.plot(column="NDVI", cmap="RdYlGn", norm=norm2, ax=axes[1], linewidth=0)
sm2 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm2)
plt.colorbar(sm2, ax=axes[1], label="NDVI (Vegetation Index)", shrink=0.8)
axes[1].set_title("Vegetation Index (NDVI)", color="white", fontsize=13, fontweight="bold")
axes[1].tick_params(colors="white")
axes[1].set_xlabel("Longitude", color="white")

# Panel C: UHI Risk Zones
risk_colors = {"Critical": "#d32f2f", "High": "#f57c00", "Moderate": "#fbc02d",
               "Low": "#388e3c", "Cool Zone": "#1565c0"}
grid["color"] = grid["risk"].map(risk_colors)
grid.plot(color=grid["color"], ax=axes[2], linewidth=0)
legend_elements = [Patch(facecolor=v, label=k) for k, v in risk_colors.items()]
axes[2].legend(handles=legend_elements, loc="lower right", framealpha=0.8,
               facecolor="#1a1a1a", labelcolor="white", fontsize=9)
axes[2].set_title("UHI Risk Classification", color="white", fontsize=13, fontweight="bold")
axes[2].tick_params(colors="white")
axes[2].set_xlabel("Longitude", color="white")

plt.suptitle(f"Urban Heat Island Analysis — {CITY_NAME}", color="white",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/uhi_static_map.png", dpi=180, bbox_inches="tight",
            facecolor="#0f0f0f")
plt.close()
print("Saved: outputs/uhi_static_map.png")

# ---------------------------------------------------------------------------
# 5. Land-use summary statistics
# ---------------------------------------------------------------------------
summary = grid.groupby("land_use").agg(
    Mean_LST=("LST", "mean"),
    Max_LST=("LST", "max"),
    Mean_NDVI=("NDVI", "mean"),
    Mean_ISF=("ISF", "mean"),
    Cell_Count=("LST", "count")
).round(2).sort_values("Mean_LST", ascending=False)

fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor("#0f0f0f")
ax2.set_facecolor("#111")
bars = ax2.barh(summary.index, summary["Mean_LST"],
                color=[cmap(norm(v)) for v in summary["Mean_LST"]])
ax2.set_xlabel("Mean Land Surface Temperature (°C)", color="white", fontsize=11)
ax2.set_title("Average LST by Land-Use Type", color="white", fontsize=13, fontweight="bold")
ax2.tick_params(colors="white")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color("#555")
for bar, val in zip(bars, summary["Mean_LST"]):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}°C", va="center", color="white", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/uhi_landuse_chart.png", dpi=150, bbox_inches="tight",
            facecolor="#0f0f0f")
plt.close()
print("Saved: outputs/uhi_landuse_chart.png")

# ---------------------------------------------------------------------------
# 6. Interactive Folium map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.045, CITY_LON + 0.045], zoom_start=12,
               tiles="CartoDB dark_matter")

heat_data = grid[["lat", "lon", "UHI_intensity"]].copy()
heat_data = heat_data[heat_data["UHI_intensity"] > 0]
HeatMap(heat_data[["lat", "lon", "UHI_intensity"]].values.tolist(),
        min_opacity=0.4, radius=18, blur=15,
        gradient={"0.2": "blue", "0.5": "yellow", "0.8": "orange", "1.0": "red"},
        name="UHI Intensity").add_to(m)

# High-risk zone markers
critical = grid[grid["risk"] == "Critical"].sample(min(20, len(grid[grid["risk"]=="Critical"])))
for _, row in critical.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6, color="#d32f2f", fill=True, fill_opacity=0.8,
        popup=folium.Popup(f"<b>CRITICAL UHI ZONE</b><br>LST: {row['LST']:.1f}°C<br>"
                           f"UHI Intensity: +{row['UHI_intensity']:.1f}°C<br>"
                           f"Land Use: {row['land_use']}<br>"
                           f"NDVI: {row['NDVI']:.2f}", max_width=250)
    ).add_to(m)

folium.LayerControl().add_to(m)
title_html = f"""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.8);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:16px;font-weight:bold;z-index:9999;">
Urban Heat Island Map — {CITY_NAME}
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("outputs/uhi_interactive_map.html")
print("Saved: outputs/uhi_interactive_map.html")

# ---------------------------------------------------------------------------
# 7. Policy recommendations
# ---------------------------------------------------------------------------
critical_count = len(grid[grid["risk"] == "Critical"])
high_count = len(grid[grid["risk"] == "High"])
total = len(grid)

print("\n" + "="*60)
print(f"  URBAN HEAT ISLAND ANALYSIS — {CITY_NAME}")
print("="*60)
print(f"  Total grid cells analyzed : {total:,}")
print(f"  Critical risk zones       : {critical_count:,} ({100*critical_count/total:.1f}%)")
print(f"  High risk zones           : {high_count:,} ({100*high_count/total:.1f}%)")
print(f"  Mean city LST             : {grid['LST'].mean():.1f}°C")
print(f"  Rural baseline LST        : {rural_mean_lst:.1f}°C")
print(f"  Peak UHI intensity        : +{grid['UHI_intensity'].max():.1f}°C")
print(f"  Mean city NDVI            : {grid['NDVI'].mean():.3f}")
print("\n  POLICY RECOMMENDATIONS:")
print("  1. Priority tree planting in Critical/High zones")
print("  2. Cool-roof mandates for new commercial construction")
print("  3. Increase green space in Industrial zones (NDVI < 0.15)")
print("  4. Urban cooling corridors along main road axes")
print("="*60)

summary.to_csv("outputs/uhi_summary_stats.csv")
print("\nSaved: outputs/uhi_summary_stats.csv")
