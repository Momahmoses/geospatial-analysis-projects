"""
Air Quality & Pollution Dispersion Mapping
Combines ground sensor readings, wind/terrain modelling, and population
exposure analysis to map PM2.5 and NO2 pollution across a city.
Identifies communities with highest health burden from air pollution.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, FancyArrow
from shapely.geometry import Point, Polygon
import folium
from folium.plugins import HeatMap
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(88)

CITY_LAT, CITY_LON = 3.8480, 11.5021   # Yaoundé, Cameroon
CITY = "Yaoundé"
GRID_ROWS, GRID_COLS = 50, 50
SPACING_KM = 0.3
lat_step = SPACING_KM / 111.32
lon_step = SPACING_KM / (111.32 * np.cos(np.radians(CITY_LAT)))

# ---------------------------------------------------------------------------
# 1. Grid setup with terrain influence
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import gaussian_filter
    topo_noise = gaussian_filter(np.random.rand(GRID_ROWS, GRID_COLS), sigma=5)
except ImportError:
    topo_noise = np.random.rand(GRID_ROWS, GRID_COLS)

center_r, center_c = 25, 25
records = []
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        lat = CITY_LAT + r * lat_step
        lon = CITY_LON + c * lon_step
        dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
        elevation = 500 + 200 * topo_noise[r, c] + 50 * np.exp(-dist/12)
        pop = max(50, 5000 * np.exp(-dist/10) + np.random.normal(0, 300))
        poly = Polygon([
            (lon, lat), (lon + lon_step, lat),
            (lon + lon_step, lat + lat_step), (lon, lat + lat_step),
        ])
        records.append({
            "row": r, "col": c, "lat": lat + lat_step/2, "lon": lon + lon_step/2,
            "elevation": elevation, "population": int(pop), "geometry": poly
        })

grid = gpd.GeoDataFrame(records, crs="EPSG:4326")

# ---------------------------------------------------------------------------
# 2. Emission sources
# ---------------------------------------------------------------------------
sources = [
    {"name": "Sonara Refinery Zone", "type": "Industrial", "lat": CITY_LAT + 0.025, "lon": CITY_LON + 0.005,
     "pm25_emission": 850, "no2_emission": 600},
    {"name": "Mvan Market", "type": "Market/Burning", "lat": CITY_LAT + 0.04, "lon": CITY_LON + 0.04,
     "pm25_emission": 320, "no2_emission": 180},
    {"name": "Industrial Zone Nord", "type": "Industrial", "lat": CITY_LAT + 0.09, "lon": CITY_LON + 0.03,
     "pm25_emission": 600, "no2_emission": 450},
    {"name": "Bus Terminal Centrale", "type": "Traffic Hub", "lat": CITY_LAT + 0.055, "lon": CITY_LON + 0.06,
     "pm25_emission": 280, "no2_emission": 520},
    {"name": "Ngoa-Ekelle Quarry", "type": "Quarry/Dust", "lat": CITY_LAT + 0.07, "lon": CITY_LON + 0.09,
     "pm25_emission": 450, "no2_emission": 90},
    {"name": "Omnisport Stadium Area", "type": "Traffic Hub", "lat": CITY_LAT + 0.06, "lon": CITY_LON + 0.055,
     "pm25_emission": 180, "no2_emission": 380},
]
sources_df = pd.DataFrame(sources)

# ---------------------------------------------------------------------------
# 3. Wind-driven dispersion model (Gaussian plume, simplified)
# ---------------------------------------------------------------------------
WIND_SPEED = 2.8   # m/s
WIND_DIR_DEG = 225  # SW wind — blowing NE
wind_u = WIND_SPEED * np.sin(np.radians(WIND_DIR_DEG))   # lon component
wind_v = WIND_SPEED * np.cos(np.radians(WIND_DIR_DEG))   # lat component

def gaussian_plume(grid_lat, grid_lon, source_lat, source_lon,
                   emission, wind_u, wind_v, sigma_y=0.012, sigma_z=0.008):
    """Simplified 2D Gaussian dispersion from a point source."""
    dx = (grid_lon - source_lon) * 111.32 * np.cos(np.radians(source_lat))
    dy = (grid_lat - source_lat) * 111.32
    dist_km = max(0.05, np.sqrt(dx**2 + dy**2))
    # Crosswind distance
    wind_mag = np.sqrt(wind_u**2 + wind_v**2)
    if wind_mag > 0:
        along = (dx * wind_u + dy * wind_v) / (wind_mag * 111.32)
        cross = (-dx * wind_v + dy * wind_u) / (wind_mag * 111.32)
    else:
        along = dist_km
        cross = 0
    if along <= 0:
        return 0  # upwind
    concentration = (emission / (2 * np.pi * sigma_y * sigma_z * wind_mag * 111.32)) * \
                    np.exp(-0.5 * (cross / sigma_y)**2) * \
                    np.exp(-0.5 * (along / (sigma_z * 5))**2)
    return max(0, concentration)

# Compute PM2.5 and NO2 concentrations
grid["pm25"] = 0.0
grid["no2"] = 0.0
for _, src in sources_df.iterrows():
    grid["pm25"] += grid.apply(
        lambda x: gaussian_plume(x["lat"], x["lon"], src["lat"], src["lon"],
                                  src["pm25_emission"], wind_u, wind_v), axis=1
    )
    grid["no2"] += grid.apply(
        lambda x: gaussian_plume(x["lat"], x["lon"], src["lat"], src["lon"],
                                  src["no2_emission"], wind_u, wind_v), axis=1
    )

# Normalise to realistic µg/m³ ranges (WHO: PM2.5 annual = 5 µg/m³)
pm25_max = grid["pm25"].max()
no2_max = grid["no2"].max()
grid["pm25_ugm3"] = 8 + 152 * (grid["pm25"] / pm25_max)  # 8–160 µg/m³
grid["no2_ugm3"] = 10 + 190 * (grid["no2"] / no2_max)    # 10–200 µg/m³

# WHO guideline thresholds
WHO_PM25 = 15   # µg/m³ annual
WHO_NO2 = 25    # µg/m³ annual

grid["pm25_exceeds"] = grid["pm25_ugm3"] > WHO_PM25
grid["no2_exceeds"] = grid["no2_ugm3"] > WHO_NO2

# AQI classification
def aqi_class(pm25):
    if pm25 <= 12: return "Good"
    elif pm25 <= 35: return "Moderate"
    elif pm25 <= 55: return "Unhealthy for Sensitive"
    elif pm25 <= 150: return "Unhealthy"
    elif pm25 <= 250: return "Very Unhealthy"
    else: return "Hazardous"

grid["aqi_class"] = grid["pm25_ugm3"].apply(aqi_class)
grid["health_burden"] = grid["pm25_ugm3"] * grid["population"] / 1e6  # burden index

# ---------------------------------------------------------------------------
# 4. Static visualisation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor("#050510")
for ax in axes.flat:
    ax.set_facecolor("#050510")

# PM2.5 map
norm_pm = mcolors.Normalize(0, 160)
grid.plot(column="pm25_ugm3", cmap="YlOrRd", norm=norm_pm, ax=axes[0][0], linewidth=0)
sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_pm)
plt.colorbar(sm, ax=axes[0][0], label="PM2.5 (µg/m³)", shrink=0.8)
for _, src in sources_df.iterrows():
    axes[0][0].scatter(src.lon, src.lat, marker="*", s=150, color="cyan",
                       zorder=5, edgecolors="white", linewidths=0.5)
# Wind arrow
axes[0][0].annotate("", xy=(CITY_LON + 0.08, CITY_LAT + 0.08),
                    xytext=(CITY_LON + 0.065, CITY_LAT + 0.065),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2))
axes[0][0].text(CITY_LON + 0.075, CITY_LAT + 0.07, f"Wind SW {WIND_SPEED}m/s",
                color="white", fontsize=8)
axes[0][0].axhline(y=CITY_LAT + 0.045, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
axes[0][0].set_title("PM2.5 Concentration (µg/m³)", color="white", fontsize=12, fontweight="bold")
axes[0][0].tick_params(colors="white")

# NO2 map
norm_no2 = mcolors.Normalize(0, 200)
grid.plot(column="no2_ugm3", cmap="PuBu", norm=norm_no2, ax=axes[0][1], linewidth=0)
sm2 = plt.cm.ScalarMappable(cmap="PuBu", norm=norm_no2)
plt.colorbar(sm2, ax=axes[0][1], label="NO2 (µg/m³)", shrink=0.8)
for _, src in sources_df.iterrows():
    axes[0][1].scatter(src.lon, src.lat, marker="*", s=120, color="yellow", zorder=5)
axes[0][1].set_title("NO2 Concentration (µg/m³)", color="white", fontsize=12, fontweight="bold")
axes[0][1].tick_params(colors="white")

# AQI classification
aqi_colors = {"Good": "#1b5e20", "Moderate": "#f9a825",
              "Unhealthy for Sensitive": "#ef6c00",
              "Unhealthy": "#b71c1c", "Very Unhealthy": "#6a1b9a", "Hazardous": "#1a237e"}
grid["aqi_color"] = grid["aqi_class"].map(aqi_colors)
grid.plot(color=grid["aqi_color"], ax=axes[1][0], linewidth=0)
legend_elements = [Patch(facecolor=v, label=k) for k, v in aqi_colors.items()]
axes[1][0].legend(handles=legend_elements, loc="upper right", framealpha=0.85,
                  facecolor="#111", labelcolor="white", fontsize=7)
axes[1][0].set_title("Air Quality Index Classification", color="white", fontsize=12, fontweight="bold")
axes[1][0].tick_params(colors="white")

# Health burden map
norm_hb = mcolors.Normalize(0, grid["health_burden"].max())
grid.plot(column="health_burden", cmap="Reds", norm=norm_hb, ax=axes[1][1], linewidth=0)
sm3 = plt.cm.ScalarMappable(cmap="Reds", norm=norm_hb)
plt.colorbar(sm3, ax=axes[1][1], label="Population Health Burden Index", shrink=0.8)
axes[1][1].set_title("Population-Weighted Health Burden", color="white", fontsize=12, fontweight="bold")
axes[1][1].tick_params(colors="white")

plt.suptitle(f"Air Quality & Pollution Dispersion — {CITY}",
             color="white", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/air_quality_map.png", dpi=180, bbox_inches="tight", facecolor="#050510")
plt.close()
print("Saved: outputs/air_quality_map.png")

# ---------------------------------------------------------------------------
# 5. Source contribution chart
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor("#050510")
for ax in axes2:
    ax.set_facecolor("#111")

src_colors = ["#d32f2f", "#f57c00", "#fdd835", "#388e3c", "#1565c0", "#6a1b9a"]
axes2[0].barh(sources_df["name"], sources_df["pm25_emission"], color=src_colors)
axes2[0].set_xlabel("PM2.5 Emission Rate (µg/s)", color="white")
axes2[0].set_title("Emission Sources — PM2.5", color="white", fontsize=12, fontweight="bold")
axes2[0].tick_params(colors="white")
for spine in axes2[0].spines.values():
    spine.set_color("#444")

axes2[1].barh(sources_df["name"], sources_df["no2_emission"], color=src_colors)
axes2[1].set_xlabel("NO2 Emission Rate (µg/s)", color="white")
axes2[1].set_title("Emission Sources — NO2", color="white", fontsize=12, fontweight="bold")
axes2[1].tick_params(colors="white")
for spine in axes2[1].spines.values():
    spine.set_color("#444")

plt.tight_layout()
plt.savefig("outputs/air_quality_sources.png", dpi=150, bbox_inches="tight", facecolor="#050510")
plt.close()
print("Saved: outputs/air_quality_sources.png")

# ---------------------------------------------------------------------------
# 6. Interactive map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.04, CITY_LON + 0.05], zoom_start=12,
               tiles="CartoDB dark_matter")

HeatMap(
    grid[["lat", "lon", "pm25_ugm3"]].values.tolist(),
    min_opacity=0.4, radius=18, blur=12,
    gradient={"0.2": "green", "0.4": "yellow", "0.6": "orange", "0.8": "red", "1.0": "purple"},
    name="PM2.5 Heatmap"
).add_to(m)

src_type_colors = {"Industrial": "red", "Market/Burning": "orange",
                   "Traffic Hub": "blue", "Quarry/Dust": "gray"}
for _, src in sources_df.iterrows():
    folium.Marker(
        location=[src.lat, src.lon],
        icon=folium.Icon(color=src_type_colors[src.type], icon="industry", prefix="fa"),
        popup=folium.Popup(
            f"<b>{src['name']}</b><br>Type: {src.type}<br>"
            f"PM2.5: {src.pm25_emission} µg/s<br>NO2: {src.no2_emission} µg/s",
            max_width=220
        ),
        tooltip=src["name"]
    ).add_to(m)

title_html = f"""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.85);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:15px;font-weight:bold;z-index:9999;">
Air Quality Map — {CITY} | PM2.5 Dispersion
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
folium.LayerControl().add_to(m)
m.save("outputs/air_quality_interactive.html")
print("Saved: outputs/air_quality_interactive.html")

# ---------------------------------------------------------------------------
# 7. Report
# ---------------------------------------------------------------------------
exposed = grid[grid["pm25_exceeds"]]["population"].sum()
total_pop = grid["population"].sum()
print("\n" + "="*60)
print(f"  AIR QUALITY REPORT — {CITY}")
print("="*60)
print(f"  Emission sources identified : {len(sources_df)}")
print(f"  Mean PM2.5 citywide        : {grid['pm25_ugm3'].mean():.1f} µg/m³ (WHO limit: {WHO_PM25})")
print(f"  Mean NO2 citywide          : {grid['no2_ugm3'].mean():.1f} µg/m³ (WHO limit: {WHO_NO2})")
print(f"  Population > WHO PM2.5 limit: {exposed:,.0f} ({100*exposed/total_pop:.1f}%)")
print(f"  Hazardous zones (AQI)       : {len(grid[grid['aqi_class']=='Hazardous']):,} cells")
print(f"  Peak PM2.5 recorded        : {grid['pm25_ugm3'].max():.1f} µg/m³")
print("\n  POLICY ACTIONS:")
print("  1. Industrial emission permits for Sonara Zone & Industrial Nord")
print("  2. Ban on open market burning in Mvan Market area")
print("  3. Dust suppression at Ngoa-Ekelle Quarry")
print("  4. Clean fuel mandate for public bus fleet")
print("  5. Real-time AQI monitors at 10 priority locations")
print("="*60)
grid.drop(columns=["geometry"]).to_csv("outputs/air_quality_data.csv", index=False)
print("\nSaved: outputs/air_quality_data.csv")
