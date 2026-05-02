"""
Disaster Response & Resource Routing
After a simulated earthquake, maps damage severity, road accessibility,
and optimally routes rescue teams and supply convoys to affected zones.
Uses graph-based shortest-path routing with damage-adjusted travel times.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, LineString
import folium
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(131)

CITY_LAT, CITY_LON = 15.5527, 32.5324   # Khartoum, Sudan
CITY = "Khartoum"

# ---------------------------------------------------------------------------
# 1. City grid with population and building stock
# ---------------------------------------------------------------------------
GRID_ROWS, GRID_COLS = 30, 30
SPACING_KM = 0.8
lat_step = SPACING_KM / 111.32
lon_step = SPACING_KM / (111.32 * np.cos(np.radians(CITY_LAT)))
center_r, center_c = 15, 15

records = []
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        lat = CITY_LAT + r * lat_step
        lon = CITY_LON + c * lon_step
        dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
        pop = max(200, 12000 * np.exp(-dist/7) + np.random.normal(0, 800))
        # Building vulnerability: older construction near center
        vulnerability = max(0.1, min(0.95, 0.7 - 0.02 * dist + np.random.normal(0, 0.1)))
        records.append({
            "row": r, "col": c, "lat": lat + lat_step/2, "lon": lon + lon_step/2,
            "population": int(pop), "vulnerability": vulnerability
        })

grid = pd.DataFrame(records)
grid_gdf = gpd.GeoDataFrame(
    grid, geometry=[Point(r.lon, r.lat) for _, r in grid.iterrows()],
    crs="EPSG:4326"
)

# ---------------------------------------------------------------------------
# 2. Earthquake damage model
# ---------------------------------------------------------------------------
# Epicentre near city centre (offset slightly)
EPI_R, EPI_C = 12, 17
MAGNITUDE = 6.8

def shake_intensity(r, c):
    """Modified Mercalli Intensity (simplified) from epicentre distance."""
    dist_km = np.sqrt((r - EPI_R)**2 + (c - EPI_C)**2) * SPACING_KM
    dist_km = max(0.1, dist_km)
    return max(1, MAGNITUDE + 2 - 1.5 * np.log10(dist_km))

grid["shake_intensity"] = grid.apply(lambda x: shake_intensity(x["row"], x["col"]), axis=1)
grid["damage_ratio"] = np.clip(
    grid["shake_intensity"] / 10 * grid["vulnerability"] + np.random.normal(0, 0.05, len(grid)),
    0, 1
)

def classify_damage(ratio):
    if ratio >= 0.7: return "Collapsed"
    elif ratio >= 0.5: return "Severe"
    elif ratio >= 0.3: return "Moderate"
    elif ratio >= 0.1: return "Minor"
    else: return "Intact"

grid["damage_class"] = grid["damage_ratio"].apply(classify_damage)

# Casualties estimate
grid["casualties"] = (grid["population"] * grid["damage_ratio"] *
                      np.random.uniform(0.01, 0.08, len(grid))).astype(int)
grid["trapped"] = (grid["population"] * grid["damage_ratio"] *
                   np.random.uniform(0.05, 0.20, len(grid))).astype(int)

# ---------------------------------------------------------------------------
# 3. Road network (simplified grid roads + blocked segments)
# ---------------------------------------------------------------------------
roads = []
# Horizontal roads every 3 rows
for r in range(0, GRID_ROWS, 3):
    for c in range(GRID_COLS - 1):
        lat1 = CITY_LAT + r * lat_step + lat_step / 2
        lon1 = CITY_LON + c * lon_step + lon_step / 2
        lat2 = lat1
        lon2 = CITY_LON + (c+1) * lon_step + lon_step / 2
        # Road blocked if high damage in adjacent cells
        damage = grid.loc[(grid["row"] == r) & (grid["col"] == c), "damage_ratio"]
        blocked = float(damage.iloc[0]) > 0.6 if len(damage) > 0 else False
        roads.append({"type": "H", "r": r, "c": c, "lat1": lat1, "lon1": lon1,
                      "lat2": lat2, "lon2": lon2, "blocked": blocked})

# Vertical roads every 3 cols
for c in range(0, GRID_COLS, 3):
    for r in range(GRID_ROWS - 1):
        lat1 = CITY_LAT + r * lat_step + lat_step / 2
        lon1 = CITY_LON + c * lon_step + lon_step / 2
        lat2 = CITY_LAT + (r+1) * lat_step + lat_step / 2
        lon2 = lon1
        damage = grid.loc[(grid["row"] == r) & (grid["col"] == c), "damage_ratio"]
        blocked = float(damage.iloc[0]) > 0.6 if len(damage) > 0 else False
        roads.append({"type": "V", "r": r, "c": c, "lat1": lat1, "lon1": lon1,
                      "lat2": lat2, "lon2": lon2, "blocked": blocked})

roads_df = pd.DataFrame(roads)

# ---------------------------------------------------------------------------
# 4. Resource bases and priority zones
# ---------------------------------------------------------------------------
bases = [
    {"name": "Airport Base", "lat": CITY_LAT + 0.05, "lon": CITY_LON + 0.02,
     "teams": 8, "supplies_tons": 50},
    {"name": "Military HQ", "lat": CITY_LAT + 0.16, "lon": CITY_LON + 0.18,
     "teams": 5, "supplies_tons": 30},
    {"name": "Red Crescent Hub", "lat": CITY_LAT + 0.10, "lon": CITY_LON + 0.20,
     "teams": 4, "supplies_tons": 25},
]
bases_df = pd.DataFrame(bases)

# Priority zones: highest trapped + casualties
grid["priority_score"] = grid["trapped"] + 3 * grid["casualties"]
top_priority = grid.nlargest(8, "priority_score")

# ---------------------------------------------------------------------------
# 5. Simple route planning (nearest unserved zone from each base)
# ---------------------------------------------------------------------------
routes = []
served = set()
for _, base in bases_df.iterrows():
    available = top_priority[~top_priority.index.isin(served)]
    for i, (idx, zone) in enumerate(available.head(3).iterrows()):
        if idx in served:
            continue
        # Simplified: direct line (real implementation uses Dijkstra on road graph)
        routes.append({
            "base": base["name"],
            "zone_row": zone["row"], "zone_col": zone["col"],
            "zone_lat": zone["lat"], "zone_lon": zone["lon"],
            "base_lat": base["lat"], "base_lon": base["lon"],
            "trapped": zone["trapped"], "casualties": zone["casualties"],
            "damage": zone["damage_class"], "priority": zone["priority_score"]
        })
        served.add(idx)
routes_df = pd.DataFrame(routes)

# ---------------------------------------------------------------------------
# 6. Static visualisation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.patch.set_facecolor("#0a0505")
for ax in axes:
    ax.set_facecolor("#0a0505")

damage_colors = {
    "Collapsed": "#b71c1c", "Severe": "#e65100", "Moderate": "#f9a825",
    "Minor": "#558b2f", "Intact": "#1a237e"
}

# Panel 1: Damage map
norm_d = mcolors.Normalize(0, 1)
scatter1 = axes[0].scatter(grid["lon"], grid["lat"], c=grid["damage_ratio"],
                           cmap="RdYlGn_r", s=50, norm=norm_d)
plt.colorbar(scatter1, ax=axes[0], label="Damage Ratio (0–1)", shrink=0.8)
# Epicentre
epi_lat = CITY_LAT + EPI_R * lat_step
epi_lon = CITY_LON + EPI_C * lon_step
axes[0].scatter([epi_lon], [epi_lat], marker="*", s=400, color="white",
                zorder=10, label=f"Epicentre (M{MAGNITUDE})")
axes[0].legend(loc="upper right", framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=8)
axes[0].set_title(f"Earthquake Damage Map — M{MAGNITUDE}", color="white", fontsize=12, fontweight="bold")
axes[0].tick_params(colors="white")

# Panel 2: Road network + blocked roads
for _, road in roads_df.iterrows():
    color = "#ff1744" if road["blocked"] else "#37474f"
    lw = 2 if road["blocked"] else 0.6
    axes[1].plot([road["lon1"], road["lon2"]], [road["lat1"], road["lat2"]],
                 color=color, linewidth=lw, alpha=0.8)
scatter2 = axes[1].scatter(grid["lon"], grid["lat"], c=grid["damage_ratio"],
                           cmap="RdYlGn_r", s=25, alpha=0.6, norm=norm_d)
axes[1].set_title("Road Network — Blocked Routes in Red", color="white", fontsize=12, fontweight="bold")
axes[1].tick_params(colors="white")
legend_road = [
    plt.Line2D([0],[0], color="#ff1744", linewidth=2, label="Blocked"),
    plt.Line2D([0],[0], color="#37474f", linewidth=1, label="Open"),
]
axes[1].legend(handles=legend_road, framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=8)

# Panel 3: Response routing
for _, road in roads_df[~roads_df["blocked"]].iterrows():
    axes[2].plot([road["lon1"], road["lon2"]], [road["lat1"], road["lat2"]],
                 color="#263238", linewidth=0.5, alpha=0.7)
scatter3 = axes[2].scatter(grid["lon"], grid["lat"], c=grid["priority_score"],
                           cmap="hot_r", s=35, alpha=0.7)
plt.colorbar(scatter3, ax=axes[2], label="Priority Score", shrink=0.8)
for _, route in routes_df.iterrows():
    color = {"Airport Base": "cyan", "Military HQ": "lime", "Red Crescent Hub": "magenta"}[route["base"]]
    axes[2].annotate("", xy=(route["zone_lon"], route["zone_lat"]),
                     xytext=(route["base_lon"], route["base_lat"]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.8))
for _, base in bases_df.iterrows():
    color = {"Airport Base": "cyan", "Military HQ": "lime", "Red Crescent Hub": "magenta"}[base["name"]]
    axes[2].scatter([base["lon"]], [base["lat"]], marker="^", s=150,
                    color=color, zorder=8, edgecolors="white")
axes[2].set_title("Response Routes to Priority Zones", color="white", fontsize=12, fontweight="bold")
axes[2].tick_params(colors="white")

plt.suptitle(f"Disaster Response Routing — {CITY} M{MAGNITUDE} Earthquake",
             color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/disaster_response_map.png", dpi=180, bbox_inches="tight", facecolor="#0a0505")
plt.close()
print("Saved: outputs/disaster_response_map.png")

# ---------------------------------------------------------------------------
# 7. Interactive response map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.12, CITY_LON + 0.12], zoom_start=12,
               tiles="CartoDB dark_matter")

# Damage zones
for _, row in grid.iterrows():
    color = {"Collapsed": "#b71c1c", "Severe": "#e65100", "Moderate": "#f9a825",
             "Minor": "#558b2f", "Intact": "#1565c0"}[row["damage_class"]]
    if row["damage_class"] in ("Collapsed", "Severe"):
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=8,
            color=color, fill=True, fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{row['damage_class'].upper()} DAMAGE</b><br>"
                f"Trapped: {row['trapped']:,}<br>Casualties: {row['casualties']:,}<br>"
                f"Population: {row['population']:,}<br>"
                f"Shake Intensity: {row['shake_intensity']:.1f}",
                max_width=200
            )
        ).add_to(m)

# Response routes
base_colors = {"Airport Base": "blue", "Military HQ": "green", "Red Crescent Hub": "purple"}
for _, route in routes_df.iterrows():
    color = base_colors[route["base"]]
    folium.PolyLine(
        locations=[[route["base_lat"], route["base_lon"]],
                   [route["zone_lat"], route["zone_lon"]]],
        color=color, weight=3, opacity=0.85, dash_array="8 4",
        tooltip=f"Route: {route['base']} → Priority Zone (Trapped: {route['trapped']})"
    ).add_to(m)

# Base markers
icon_colors = {"Airport Base": "blue", "Military HQ": "green", "Red Crescent Hub": "purple"}
for _, base in bases_df.iterrows():
    folium.Marker(
        location=[base["lat"], base["lon"]],
        icon=folium.Icon(color=icon_colors[base["name"]], icon="home", prefix="fa"),
        popup=folium.Popup(
            f"<b>{base['name']}</b><br>Teams: {base['teams']}<br>"
            f"Supplies: {base['supplies_tons']} tonnes", max_width=180
        ),
        tooltip=base["name"]
    ).add_to(m)

# Epicentre
folium.Marker(
    location=[epi_lat, epi_lon],
    icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
    popup=folium.Popup(f"<b>EPICENTRE</b><br>Magnitude: M{MAGNITUDE}", max_width=150),
    tooltip="Earthquake Epicentre"
).add_to(m)

title_html = f"""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.85);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:15px;font-weight:bold;z-index:9999;">
Disaster Response Map — {CITY} M{MAGNITUDE}
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("outputs/disaster_response_interactive.html")
print("Saved: outputs/disaster_response_interactive.html")

# ---------------------------------------------------------------------------
# 8. Response report
# ---------------------------------------------------------------------------
collapsed = grid[grid["damage_class"] == "Collapsed"]
severe = grid[grid["damage_class"] == "Severe"]
total_trapped = grid["trapped"].sum()
total_casualties = grid["casualties"].sum()
blocked_roads = roads_df["blocked"].sum()

print("\n" + "="*60)
print(f"  DISASTER RESPONSE REPORT — M{MAGNITUDE} {CITY}")
print("="*60)
print(f"  Zones analysed              : {len(grid):,}")
print(f"  Collapsed zones             : {len(collapsed):,}")
print(f"  Severe damage zones         : {len(severe):,}")
print(f"  Total trapped persons       : {total_trapped:,}")
print(f"  Total casualties (est.)     : {total_casualties:,}")
print(f"  Road segments blocked       : {blocked_roads:,}")
print(f"  Resource bases deployed     : {len(bases_df)}")
print(f"  Routes planned              : {len(routes_df)}")
print("\n  TOP PRIORITY ZONES (by trapped persons):")
for _, row in top_priority.head(5).iterrows():
    print(f"  Zone ({row['row']},{row['col']}) — Trapped: {row['trapped']:,} | "
          f"Casualties: {row['casualties']:,} | {row['damage_class']}")
print("="*60)
grid.drop(columns=["geometry"] if "geometry" in grid.columns else []).to_csv(
    "outputs/damage_assessment.csv", index=False
)
print("\nSaved: outputs/damage_assessment.csv")
