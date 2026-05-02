"""
Public Transit Gap Analysis
Analyses commute patterns vs transit route coverage to identify underserved
corridors. Guides bus route investments that reduce car dependency and
improve mobility for low-income residents.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, LineString, Polygon
import folium
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(161)

CITY_LAT, CITY_LON = -1.2921, 36.8219   # Nairobi, Kenya
CITY = "Nairobi"
GRID_ROWS, GRID_COLS = 40, 40
SPACING_KM = 0.6
lat_step = SPACING_KM / 111.32
lon_step = SPACING_KM / (111.32 * np.cos(np.radians(CITY_LAT)))
center_r, center_c = 20, 20

# ---------------------------------------------------------------------------
# 1. Population and employment grid
# ---------------------------------------------------------------------------
records = []
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        lat = CITY_LAT + r * lat_step
        lon = CITY_LON + c * lon_step
        dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
        # Population density — dense residential ring 5-15 units from centre
        ring_peak = max(0, 1 - abs(dist - 10) / 10)
        pop = max(200, 4000 * ring_peak + 1500 * np.exp(-dist/6) + np.random.normal(0, 300))
        # Employment — concentrated at centre + 2 business parks
        jobs = max(0, 8000 * np.exp(-dist/5) +
                   2000 * np.exp(-np.sqrt((r-8)**2 + (c-30)**2)/4) +
                   1500 * np.exp(-np.sqrt((r-32)**2 + (c-12)**2)/4) +
                   np.random.normal(0, 200))
        income_index = max(10, 75 - 2 * dist + np.random.normal(0, 8))
        car_ownership = min(0.9, max(0.02, 0.05 + 0.008 * income_index + np.random.normal(0, 0.04)))
        poly = Polygon([
            (lon, lat), (lon + lon_step, lat),
            (lon + lon_step, lat + lat_step), (lon, lat + lat_step),
        ])
        records.append({
            "row": r, "col": c, "lat": lat + lat_step/2, "lon": lon + lon_step/2,
            "population": int(pop), "jobs": int(jobs),
            "income_index": round(income_index, 1),
            "car_ownership": round(car_ownership, 3),
            "geometry": poly
        })

grid = gpd.GeoDataFrame(records, crs="EPSG:4326")

# ---------------------------------------------------------------------------
# 2. Existing transit routes (matatu lines — Nairobi style)
# ---------------------------------------------------------------------------
transit_routes = [
    {"name": "Route 2 — CBD–Westlands", "color": "#e53935",
     "stops": [(center_r, c) for c in range(5, 30, 2)]},
    {"name": "Route 46 — CBD–Eastleigh", "color": "#fb8c00",
     "stops": [(r, center_c) for r in range(5, 35, 2)]},
    {"name": "Route 58 — CBD–Kibera", "color": "#43a047",
     "stops": [(center_r - i, center_c - i) for i in range(0, 15, 2)]},
    {"name": "Route 111 — Thika Road", "color": "#1e88e5",
     "stops": [(r, center_c + 5) for r in range(0, 35, 2)]},
    {"name": "Route 34 — Lang'ata", "color": "#8e24aa",
     "stops": [(center_r + i, center_c - i//2) for i in range(0, 18, 2)]},
]

# Convert stops to geographic coords
route_gdfs = []
for route in transit_routes:
    stop_lats = [CITY_LAT + s[0] * lat_step + lat_step/2 for s in route["stops"]]
    stop_lons = [CITY_LON + s[1] * lon_step + lon_step/2 for s in route["stops"]]
    route_gdfs.append({
        "name": route["name"], "color": route["color"],
        "lats": stop_lats, "lons": stop_lons,
        "geometry": LineString(zip(stop_lons, stop_lats))
    })

routes_gdf = gpd.GeoDataFrame(route_gdfs, crs="EPSG:4326")

# ---------------------------------------------------------------------------
# 3. Transit accessibility score per cell
# ---------------------------------------------------------------------------
def transit_access(grid_lat, grid_lon, car_rate):
    """Score based on proximity to transit stops."""
    best_dist = float("inf")
    for route in transit_routes:
        for stop_r, stop_c in route["stops"]:
            stop_lat = CITY_LAT + stop_r * lat_step + lat_step / 2
            stop_lon = CITY_LON + stop_c * lon_step + lon_step / 2
            dx = (grid_lon - stop_lon) * 111.32 * np.cos(np.radians(grid_lat))
            dy = (grid_lat - stop_lat) * 111.32
            dist = np.sqrt(dx**2 + dy**2)
            best_dist = min(best_dist, dist)
    # Score: within 0.5km walk → full, 1km → half, >2km → low
    walk_score = max(0, 1 - best_dist / 2.0)
    return walk_score * 100

grid["transit_score"] = grid.apply(
    lambda x: transit_access(x["lat"], x["lon"], x["car_ownership"]), axis=1
)

# Demand score: population × (1 - car_ownership) × commute need
# (proxy: jobs nearby drives commute demand)
grid_lons = grid["lon"].values
grid_lats = grid["lat"].values
grid_jobs = grid["jobs"].values

demand_vals = np.zeros(len(grid))
for i in range(len(grid)):
    dx = (grid_lons - grid_lons[i]) * 111.32 * np.cos(np.radians(grid_lats[i]))
    dy = (grid_lats - grid_lats[i]) * 111.32
    dist_km = np.maximum(0.1, np.sqrt(dx**2 + dy**2))
    mask = dist_km < 8
    job_access = np.sum(grid_jobs[mask] / dist_km[mask])
    demand_vals[i] = grid["population"].iloc[i] * (1 - grid["car_ownership"].iloc[i]) * np.log1p(job_access) / 1000

grid["demand"] = demand_vals
grid["demand"] = (grid["demand"] - grid["demand"].min()) / (grid["demand"].max() - grid["demand"].min()) * 100

# Transit gap = high demand but low transit access
grid["gap_score"] = grid["demand"] * (1 - grid["transit_score"] / 100)

def classify_gap(gap):
    if gap >= 60: return "Critical Gap"
    elif gap >= 40: return "High Gap"
    elif gap >= 20: return "Moderate Gap"
    elif gap >= 5: return "Low Gap"
    else: return "Well Served"

grid["gap_class"] = grid["gap_score"].apply(classify_gap)

# ---------------------------------------------------------------------------
# 4. Identify top candidate corridors for new routes
# ---------------------------------------------------------------------------
critical_gaps = grid[grid["gap_class"] == "Critical Gap"]
top_gap_zones = grid.nlargest(10, "gap_score")

# ---------------------------------------------------------------------------
# 5. Static visualisation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor("#0a0a10")
for ax in axes.flat:
    ax.set_facecolor("#0a0a10")

# Population + transit routes
norm_p = mcolors.Normalize(vmin=200, vmax=grid["population"].max())
grid.plot(column="population", cmap="YlOrRd", norm=norm_p, ax=axes[0][0], linewidth=0)
sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_p)
plt.colorbar(sm, ax=axes[0][0], label="Population", shrink=0.8)
for route in transit_routes:
    xs = [CITY_LON + s[1] * lon_step + lon_step/2 for s in route["stops"]]
    ys = [CITY_LAT + s[0] * lat_step + lat_step/2 for s in route["stops"]]
    axes[0][0].plot(xs, ys, color=route["color"], linewidth=2.5, alpha=0.85, label=route["name"][:15])
axes[0][0].set_title("Population Density + Transit Routes", color="white", fontsize=11, fontweight="bold")
axes[0][0].tick_params(colors="white")
axes[0][0].legend(loc="upper right", framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=6)

# Transit access score
norm_t = mcolors.Normalize(0, 100)
grid.plot(column="transit_score", cmap="RdYlGn", norm=norm_t, ax=axes[0][1], linewidth=0)
sm2 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm_t)
plt.colorbar(sm2, ax=axes[0][1], label="Transit Access Score (0–100)", shrink=0.8)
axes[0][1].set_title("Transit Accessibility Score", color="white", fontsize=11, fontweight="bold")
axes[0][1].tick_params(colors="white")

# Demand map
norm_d = mcolors.Normalize(0, 100)
grid.plot(column="demand", cmap="Blues", norm=norm_d, ax=axes[1][0], linewidth=0)
sm3 = plt.cm.ScalarMappable(cmap="Blues", norm=norm_d)
plt.colorbar(sm3, ax=axes[1][0], label="Transit Demand Score (0–100)", shrink=0.8)
axes[1][0].set_title("Transit Demand (Pop × No-Car × Jobs)", color="white", fontsize=11, fontweight="bold")
axes[1][0].tick_params(colors="white")

# Gap map
gap_colors = {
    "Critical Gap": "#880e4f", "High Gap": "#c62828",
    "Moderate Gap": "#f57c00", "Low Gap": "#f9a825", "Well Served": "#1b5e20"
}
grid["gap_color"] = grid["gap_class"].map(gap_colors)
grid.plot(color=grid["gap_color"], ax=axes[1][1], linewidth=0)
for route in transit_routes:
    xs = [CITY_LON + s[1] * lon_step + lon_step/2 for s in route["stops"]]
    ys = [CITY_LAT + s[0] * lat_step + lat_step/2 for s in route["stops"]]
    axes[1][1].plot(xs, ys, color=route["color"], linewidth=2, alpha=0.7)
# Mark top gap zones
axes[1][1].scatter(top_gap_zones["lon"], top_gap_zones["lat"],
                   marker="*", s=150, color="white", zorder=6, label="Proposed Routes")
legend_elements = [Patch(facecolor=v, label=k) for k, v in gap_colors.items()]
legend_elements.append(plt.Line2D([0],[0], marker="*", color="white",
                                   linestyle="None", markersize=8, label="New Route Candidate"))
axes[1][1].legend(handles=legend_elements, loc="lower right", framealpha=0.8,
                  facecolor="#111", labelcolor="white", fontsize=7)
axes[1][1].set_title("Transit Gap Classification + Route Candidates", color="white", fontsize=11, fontweight="bold")
axes[1][1].tick_params(colors="white")

plt.suptitle(f"Public Transit Gap Analysis — {CITY}", color="white",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/transit_gap_map.png", dpi=180, bbox_inches="tight", facecolor="#0a0a10")
plt.close()
print("Saved: outputs/transit_gap_map.png")

# ---------------------------------------------------------------------------
# 6. Commuter mode share chart
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor("#0a0a10")
for ax in axes2:
    ax.set_facecolor("#111")

by_gap = grid.groupby("gap_class")["population"].sum().reindex(
    ["Critical Gap", "High Gap", "Moderate Gap", "Low Gap", "Well Served"]
).fillna(0)
gap_colors_list = [gap_colors[k] for k in by_gap.index]
axes2[0].barh(by_gap.index, by_gap.values / 1000, color=gap_colors_list)
axes2[0].set_xlabel("Population (thousands)", color="white")
axes2[0].set_title("Population by Transit Gap Level", color="white", fontsize=12, fontweight="bold")
axes2[0].tick_params(colors="white")
for spine in axes2[0].spines.values():
    spine.set_color("#444")

# Income vs transit score scatter
axes2[1].scatter(grid["income_index"], grid["transit_score"],
                 c=grid["gap_score"], cmap="RdYlGn_r", s=15, alpha=0.6)
sm4 = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                              norm=mcolors.Normalize(0, grid["gap_score"].max()))
plt.colorbar(sm4, ax=axes2[1], label="Gap Score", shrink=0.8)
axes2[1].axhline(y=30, color="white", linestyle="--", linewidth=0.8, alpha=0.7,
                  label="Low access threshold (30)")
axes2[1].axvline(x=30, color="yellow", linestyle="--", linewidth=0.8, alpha=0.7,
                  label="Low income threshold (30)")
axes2[1].set_xlabel("Income Index", color="white")
axes2[1].set_ylabel("Transit Access Score", color="white")
axes2[1].set_title("Income vs Transit Access (Gap = colour)", color="white", fontsize=12, fontweight="bold")
axes2[1].tick_params(colors="white")
axes2[1].legend(framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=8)
for spine in axes2[1].spines.values():
    spine.set_color("#444")

plt.tight_layout()
plt.savefig("outputs/transit_gap_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0a0a10")
plt.close()
print("Saved: outputs/transit_gap_analysis.png")

# ---------------------------------------------------------------------------
# 7. Interactive map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.03, CITY_LON + 0.03], zoom_start=12,
               tiles="CartoDB dark_matter")

for _, row in grid[grid["gap_class"].isin(["Critical Gap", "High Gap"])].iterrows():
    color = gap_colors[row["gap_class"]]
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=6,
        color=color, fill=True, fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{row['gap_class']}</b><br>"
            f"Gap Score: {row['gap_score']:.1f}<br>"
            f"Pop: {row['population']:,}<br>"
            f"Car Ownership: {100*row['car_ownership']:.0f}%<br>"
            f"Income Index: {row['income_index']:.0f}<br>"
            f"Transit Score: {row['transit_score']:.1f}/100",
            max_width=200
        )
    ).add_to(m)

for route in transit_routes:
    latlons = [(CITY_LAT + s[0] * lat_step + lat_step/2,
                CITY_LON + s[1] * lon_step + lon_step/2) for s in route["stops"]]
    folium.PolyLine(
        locations=latlons, color=route["color"], weight=3,
        opacity=0.85, tooltip=route["name"]
    ).add_to(m)
    for lat, lon in latlons[::2]:
        folium.CircleMarker(
            location=[lat, lon], radius=4, color=route["color"],
            fill=True, fill_opacity=0.9
        ).add_to(m)

# Proposed new route candidates
for _, row in top_gap_zones.head(5).iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.Icon(color="white", icon="star", prefix="fa"),
        popup=folium.Popup(
            f"<b>NEW ROUTE CANDIDATE</b><br>"
            f"Gap Score: {row['gap_score']:.1f}<br>"
            f"Population: {row['population']:,}<br>"
            f"No-Car Commuters: {int(row['population'] * (1-row['car_ownership'])):,}",
            max_width=220
        ),
        tooltip="Proposed Route"
    ).add_to(m)

title_html = f"""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.85);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:15px;font-weight:bold;z-index:9999;">
Public Transit Gap Analysis — {CITY}
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("outputs/transit_gap_interactive.html")
print("Saved: outputs/transit_gap_interactive.html")

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------
critical_pop = grid[grid["gap_class"] == "Critical Gap"]["population"].sum()
high_pop = grid[grid["gap_class"] == "High Gap"]["population"].sum()
total_pop = grid["population"].sum()
no_car_critical = (grid[grid["gap_class"] == "Critical Gap"]["population"] *
                   (1 - grid[grid["gap_class"] == "Critical Gap"]["car_ownership"])).sum()

print("\n" + "="*60)
print(f"  TRANSIT GAP ANALYSIS — {CITY}")
print("="*60)
print(f"  Total population             : {total_pop:,.0f}")
print(f"  Critical gap population      : {critical_pop:,.0f} ({100*critical_pop/total_pop:.1f}%)")
print(f"  High gap population          : {high_pop:,.0f} ({100*high_pop/total_pop:.1f}%)")
print(f"  No-car commuters in critical zones: {int(no_car_critical):,}")
print(f"  Existing routes              : {len(transit_routes)}")
print(f"  New route candidates         : {len(top_gap_zones)}")
print("\n  TOP 5 NEW ROUTE CANDIDATES (by gap score):")
for i, (_, row) in enumerate(top_gap_zones.head(5).iterrows(), 1):
    no_car = int(row["population"] * (1 - row["car_ownership"]))
    print(f"  {i}. Zone ({row['row']},{row['col']}) — Gap: {row['gap_score']:.1f} | "
          f"Pop: {row['population']:,} | No-car commuters: {no_car:,}")
print("\n  RECOMMENDATIONS:")
print("  1. New east-west corridor through Critical Gap zones")
print("  2. BRT feeder service to Thika Road from gap clusters")
print("  3. Frequency boost on Route 58 (Kibera — highest no-car demand)")
print("  4. First/last mile micro-transit in southern suburbs")
print("="*60)
grid.drop(columns=["geometry"]).to_csv("outputs/transit_gap_data.csv", index=False)
print("\nSaved: outputs/transit_gap_data.csv")
