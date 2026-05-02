"""
Road Safety & Accident Hotspot Analysis
Maps traffic accident data to identify dangerous intersections and road segments.
Uses Kernel Density Estimation (KDE) and spatial clustering to produce
actionable hotspot maps for infrastructure improvement.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, LineString, MultiPoint
from sklearn.neighbors import KernelDensity
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(77)

CITY_LAT, CITY_LON = 6.3703, 2.3912   # Cotonou, Benin Republic
CITY = "Cotonou"

# ---------------------------------------------------------------------------
# 1. Road network — major arteries and intersections
# ---------------------------------------------------------------------------
major_roads = [
    {"name": "Boulevard de la Marina", "type": "Arterial",
     "coords": [(CITY_LON + i*0.005, CITY_LAT + 0.02) for i in range(20)]},
    {"name": "Avenue Steinmetz", "type": "Arterial",
     "coords": [(CITY_LON + 0.05, CITY_LAT + i*0.004) for i in range(20)]},
    {"name": "Route de l'Aéroport", "type": "Highway",
     "coords": [(CITY_LON + i*0.006, CITY_LAT + 0.05 + i*0.002) for i in range(15)]},
    {"name": "Boulevard Saint Michel", "type": "Arterial",
     "coords": [(CITY_LON + i*0.004, CITY_LAT + 0.035) for i in range(20)]},
    {"name": "Rue du Gouverneur Général", "type": "Secondary",
     "coords": [(CITY_LON + 0.03 + i*0.003, CITY_LAT + i*0.003) for i in range(15)]},
    {"name": "Avenue Jean Paul II", "type": "Arterial",
     "coords": [(CITY_LON + 0.07, CITY_LAT + i*0.004) for i in range(20)]},
]

roads_gdf = gpd.GeoDataFrame(
    [{"name": r["name"], "type": r["type"],
      "geometry": LineString(r["coords"])} for r in major_roads],
    crs="EPSG:4326"
)

# Major intersection points
intersections = [
    {"id": "INT-01", "name": "Marina/Steinmetz", "lat": CITY_LAT + 0.02, "lon": CITY_LON + 0.05},
    {"id": "INT-02", "name": "Marina/JP2", "lat": CITY_LAT + 0.02, "lon": CITY_LON + 0.07},
    {"id": "INT-03", "name": "St Michel/Steinmetz", "lat": CITY_LAT + 0.035, "lon": CITY_LON + 0.05},
    {"id": "INT-04", "name": "Airport Rd/JP2", "lat": CITY_LAT + 0.065, "lon": CITY_LON + 0.07},
    {"id": "INT-05", "name": "GG/Steinmetz", "lat": CITY_LAT + 0.04, "lon": CITY_LON + 0.06},
]

# ---------------------------------------------------------------------------
# 2. Generate realistic accident dataset (5 years, 2020–2024)
# ---------------------------------------------------------------------------
N_ACCIDENTS = 1200

# Accidents cluster near intersections and busy roads
accident_records = []
intersection_lats = [i["lat"] for i in intersections]
intersection_lons = [i["lon"] for i in intersections]

for _ in range(N_ACCIDENTS):
    r = np.random.rand()
    if r < 0.55:   # cluster near intersections (high density)
        idx = np.random.randint(len(intersections))
        lat = intersection_lats[idx] + np.random.normal(0, 0.003)
        lon = intersection_lons[idx] + np.random.normal(0, 0.003)
    elif r < 0.80:  # along road corridors
        road = np.random.randint(len(major_roads))
        coords = major_roads[road]["coords"]
        pt = coords[np.random.randint(len(coords))]
        lon = pt[0] + np.random.normal(0, 0.002)
        lat = pt[1] + np.random.normal(0, 0.002)
    else:           # random citywide
        lat = CITY_LAT + np.random.uniform(0, 0.08)
        lon = CITY_LON + np.random.uniform(0, 0.09)

    # Severity
    sev_roll = np.random.rand()
    if sev_roll < 0.06:
        severity = "Fatal"
        casualties = np.random.randint(1, 4)
        injuries = np.random.randint(0, 3)
    elif sev_roll < 0.25:
        severity = "Serious Injury"
        casualties = 0
        injuries = np.random.randint(1, 4)
    elif sev_roll < 0.65:
        severity = "Minor Injury"
        casualties = 0
        injuries = np.random.randint(1, 3)
    else:
        severity = "Property Damage"
        casualties = 0
        injuries = 0

    year = np.random.randint(2020, 2025)
    hour = int(np.random.choice(range(24), p=[
        0.01,0.01,0.01,0.01,0.01,0.02,0.04,0.07,0.07,0.05,0.04,0.05,
        0.06,0.05,0.04,0.05,0.07,0.08,0.07,0.06,0.05,0.04,0.03,0.01]))
    vehicle_type = np.random.choice(
        ["Motorcycle", "Car", "Bus/Truck", "Bicycle"],
        p=[0.45, 0.35, 0.12, 0.08]
    )
    accident_records.append({
        "lat": round(lat, 6), "lon": round(lon, 6),
        "severity": severity, "year": year, "hour": hour,
        "casualties": casualties, "injuries": injuries,
        "vehicle_type": vehicle_type,
        "day_type": "Weekend" if np.random.rand() < 0.28 else "Weekday"
    })

accidents = pd.DataFrame(accident_records)
accidents_gdf = gpd.GeoDataFrame(
    accidents, geometry=[Point(r.lon, r.lat) for _, r in accidents.iterrows()],
    crs="EPSG:4326"
)

# ---------------------------------------------------------------------------
# 3. Kernel Density Estimation for hotspot map
# ---------------------------------------------------------------------------
coords = accidents_gdf[["lon", "lat"]].values
kde = KernelDensity(bandwidth=0.004, metric="haversine", kernel="gaussian")
kde.fit(np.radians(coords[:, ::-1]))   # lat, lon order for haversine

# Grid for KDE evaluation
lat_range = np.linspace(CITY_LAT, CITY_LAT + 0.09, 100)
lon_range = np.linspace(CITY_LON, CITY_LON + 0.10, 100)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
grid_pts = np.radians(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))
log_density = kde.score_samples(grid_pts)
density = np.exp(log_density).reshape(100, 100)

# ---------------------------------------------------------------------------
# 4. Hotspot classification
# ---------------------------------------------------------------------------
p75 = np.percentile(density, 75)
p90 = np.percentile(density, 90)
p95 = np.percentile(density, 95)

hotspot_class = np.where(density >= p95, "Black Spot",
                np.where(density >= p90, "High Risk",
                np.where(density >= p75, "Elevated", "Normal")))

# ---------------------------------------------------------------------------
# 5. Static visualisation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor("#0a0a0a")
for ax in axes.flat:
    ax.set_facecolor("#0a0a0a")

sev_colors = {"Fatal": "#b71c1c", "Serious Injury": "#e65100",
              "Minor Injury": "#f9a825", "Property Damage": "#1565c0"}

# KDE heatmap
extent = [CITY_LON, CITY_LON + 0.10, CITY_LAT, CITY_LAT + 0.09]
im = axes[0][0].imshow(density, extent=extent, origin="lower",
                       cmap="hot", aspect="auto", alpha=0.85)
plt.colorbar(im, ax=axes[0][0], label="Accident Density", shrink=0.8)
for _, road in roads_gdf.iterrows():
    xs = [p[0] for p in road.geometry.coords]
    ys = [p[1] for p in road.geometry.coords]
    axes[0][0].plot(xs, ys, color="cyan", linewidth=0.8, alpha=0.6)
axes[0][0].set_title("Accident Density Heatmap (KDE)", color="white", fontsize=12, fontweight="bold")
axes[0][0].tick_params(colors="white")

# Severity scatter
for sev, color in sev_colors.items():
    sub = accidents_gdf[accidents_gdf["severity"] == sev]
    axes[0][1].scatter(sub["lon"], sub["lat"], c=color, s=8,
                       alpha=0.6, label=sev, zorder=3)
for _, road in roads_gdf.iterrows():
    xs = [p[0] for p in road.geometry.coords]
    ys = [p[1] for p in road.geometry.coords]
    axes[0][1].plot(xs, ys, color="#444", linewidth=0.8)
axes[0][1].legend(loc="upper right", framealpha=0.8, facecolor="#111",
                  labelcolor="white", fontsize=8, markerscale=2)
axes[0][1].set_title("Accidents by Severity", color="white", fontsize=12, fontweight="bold")
axes[0][1].tick_params(colors="white")

# Hourly distribution
hourly = accidents.groupby(["hour", "severity"]).size().unstack(fill_value=0)
hours = list(range(24))
bottom = np.zeros(24)
for sev, color in sev_colors.items():
    if sev in hourly.columns:
        vals = hourly.reindex(hours).fillna(0)[sev].values
        axes[1][0].bar(hours, vals, bottom=bottom, color=color, label=sev, alpha=0.85)
        bottom += vals
axes[1][0].set_xlabel("Hour of Day", color="white")
axes[1][0].set_ylabel("Number of Accidents", color="white")
axes[1][0].set_title("Accidents by Hour of Day", color="white", fontsize=12, fontweight="bold")
axes[1][0].tick_params(colors="white")
axes[1][0].legend(framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=8)
for spine in axes[1][0].spines.values():
    spine.set_color("#444")

# Vehicle type chart
veh_counts = accidents.groupby(["vehicle_type", "severity"]).size().unstack(fill_value=0)
veh_types = veh_counts.index.tolist()
x = np.arange(len(veh_types))
width = 0.2
for i, (sev, color) in enumerate(sev_colors.items()):
    if sev in veh_counts.columns:
        axes[1][1].bar(x + i*width, veh_counts[sev], width, color=color, label=sev, alpha=0.85)
axes[1][1].set_xticks(x + width*1.5)
axes[1][1].set_xticklabels(veh_types, color="white", rotation=10)
axes[1][1].set_ylabel("Accidents", color="white")
axes[1][1].set_title("Accidents by Vehicle Type & Severity", color="white", fontsize=12, fontweight="bold")
axes[1][1].tick_params(colors="white")
axes[1][1].legend(framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=8)
for spine in axes[1][1].spines.values():
    spine.set_color("#444")

plt.suptitle(f"Road Safety Hotspot Analysis — {CITY}", color="white",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/road_safety_analysis.png", dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print("Saved: outputs/road_safety_analysis.png")

# ---------------------------------------------------------------------------
# 6. Interactive hotspot map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.04, CITY_LON + 0.05], zoom_start=13,
               tiles="CartoDB dark_matter")

heat_data = [[r.lat, r.lon] for _, r in accidents_gdf.iterrows()]
HeatMap(heat_data, min_opacity=0.4, radius=20, blur=15,
        gradient={"0.2": "blue", "0.5": "yellow", "0.8": "orange", "1.0": "red"},
        name="Accident Density").add_to(m)

cluster = MarkerCluster(name="Individual Accidents").add_to(m)
for _, row in accidents_gdf[accidents_gdf["severity"].isin(["Fatal", "Serious Injury"])].iterrows():
    color = "#b71c1c" if row["severity"] == "Fatal" else "#e65100"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=6,
        color=color, fill=True, fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['severity']}</b><br>"
            f"Year: {row['year']} | Hour: {row['hour']}:00<br>"
            f"Vehicle: {row['vehicle_type']}<br>"
            f"Casualties: {row['casualties']} | Injuries: {row['injuries']}",
            max_width=200
        )
    ).add_to(cluster)

for road_data in major_roads:
    folium.PolyLine(
        locations=[(c[1], c[0]) for c in road_data["coords"]],
        color="cyan", weight=1.5, opacity=0.5, tooltip=road_data["name"]
    ).add_to(m)

folium.LayerControl().add_to(m)
title_html = f"""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.85);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:15px;font-weight:bold;z-index:9999;">
Road Safety Hotspot Map — {CITY}
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("outputs/road_safety_interactive.html")
print("Saved: outputs/road_safety_interactive.html")

# ---------------------------------------------------------------------------
# 7. Report
# ---------------------------------------------------------------------------
fatal = len(accidents[accidents["severity"] == "Fatal"])
serious = len(accidents[accidents["severity"] == "Serious Injury"])
print("\n" + "="*60)
print(f"  ROAD SAFETY REPORT — {CITY} (2020–2024)")
print("="*60)
print(f"  Total accidents        : {len(accidents):,}")
print(f"  Fatal accidents        : {fatal:,} ({100*fatal/len(accidents):.1f}%)")
print(f"  Serious injuries       : {serious:,} ({100*serious/len(accidents):.1f}%)")
print(f"  Total casualties       : {accidents['casualties'].sum():,}")
print(f"  Total injuries         : {accidents['injuries'].sum():,}")
print(f"  Peak hour              : {accidents['hour'].value_counts().idxmax()}:00")
print(f"  Most dangerous vehicle : {accidents['vehicle_type'].value_counts().index[0]}")
print("\n  TOP HOTSPOT INTERSECTIONS:")
for inter in intersections[:5]:
    nearby = accidents[
        (np.sqrt((accidents.lat - inter["lat"])**2 + (accidents.lon - inter["lon"])**2)) < 0.005
    ]
    print(f"  • {inter['name']}: {len(nearby)} accidents")
print("\n  RECOMMENDED INTERVENTIONS:")
print("  1. Roundabout conversion at top 3 signalised intersections")
print("  2. Motorcycle speed enforcement 7–9am and 5–8pm")
print("  3. Road markings and lighting upgrade on Marina corridor")
print("  4. Pedestrian crossing infrastructure at identified black spots")
print("="*60)
accidents.to_csv("outputs/accident_data.csv", index=False)
print("\nSaved: outputs/accident_data.csv")
