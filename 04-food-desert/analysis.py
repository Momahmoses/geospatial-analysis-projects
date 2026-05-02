"""
Food Desert Identification
Overlays grocery store locations, income levels, and transportation networks
to identify communities with poor access to fresh, healthy food.
Uses USDA-style food desert criteria adapted for urban African context.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
import folium
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(33)

CITY_LAT, CITY_LON = 5.5600, -0.2057   # Accra, Ghana
GRID_ROWS, GRID_COLS = 45, 45
SPACING_KM = 0.4
lat_step = SPACING_KM / 111.32
lon_step = SPACING_KM / (111.32 * np.cos(np.radians(CITY_LAT)))
center_r, center_c = 22, 22

# ---------------------------------------------------------------------------
# 1. Population & income grid
# ---------------------------------------------------------------------------
records = []
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        lat = CITY_LAT + r * lat_step
        lon = CITY_LON + c * lon_step
        dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
        pop = max(100, 6000 * np.exp(-dist/9) + np.random.normal(0, 500))
        # Income index 0-100: higher near centre and northern suburbs
        income = max(5, 85 * np.exp(-dist/14) + np.random.normal(0, 8))
        # Vehicle ownership rate (correlated with income)
        vehicle_rate = min(0.95, max(0.02, 0.1 + 0.008 * income + np.random.normal(0, 0.05)))
        poly = Polygon([
            (lon, lat), (lon + lon_step, lat),
            (lon + lon_step, lat + lat_step), (lon, lat + lat_step),
        ])
        records.append({
            "row": r, "col": c,
            "lat": lat + lat_step/2, "lon": lon + lon_step/2,
            "population": int(pop), "income_index": round(income, 1),
            "vehicle_rate": round(vehicle_rate, 3), "geometry": poly
        })

grid = gpd.GeoDataFrame(records, crs="EPSG:4326")

# ---------------------------------------------------------------------------
# 2. Grocery store / fresh food outlet locations
#    Biased toward wealthy/central areas — creating food deserts in periphery
# ---------------------------------------------------------------------------
n_stores = 35
store_lats, store_lons, store_types, store_names = [], [], [], []

store_type_names = {
    "Supermarket": ["Shoprite", "MaxMart", "Palace Mall", "A&C Mall", "Junction Mall"],
    "Market": ["Agbogbloshie", "Makola", "Kaneshie", "Dome", "Spintex"],
    "Mini-Mart": ["QuickShop", "SaveMore", "FreshStop", "GreenBasket", "NutriMart"],
}

for i in range(n_stores):
    prob_central = 0.6  # 60% of stores in central half
    if np.random.rand() < prob_central:
        r = int(np.clip(np.random.normal(center_r, 7), 5, GRID_ROWS-5))
        c = int(np.clip(np.random.normal(center_c, 7), 5, GRID_COLS-5))
    else:
        r = int(np.random.randint(2, GRID_ROWS-2))
        c = int(np.random.randint(2, GRID_COLS-2))

    store_lats.append(CITY_LAT + r * lat_step)
    store_lons.append(CITY_LON + c * lon_step)

    dist_from_center = np.sqrt((r-center_r)**2 + (c-center_c)**2)
    if dist_from_center < 8:
        stype = np.random.choice(["Supermarket", "Market", "Mini-Mart"], p=[0.4, 0.3, 0.3])
    else:
        stype = np.random.choice(["Market", "Mini-Mart"], p=[0.5, 0.5])
    store_types.append(stype)
    store_names.append(np.random.choice(store_type_names[stype]))

stores = pd.DataFrame({
    "name": store_names, "type": store_types,
    "lat": store_lats, "lon": store_lons,
    "fresh_score": [np.random.randint(60, 100) if t == "Supermarket"
                    else (np.random.randint(40, 85) if t == "Market"
                    else np.random.randint(20, 60)) for t in store_types]
})
stores_gdf = gpd.GeoDataFrame(
    stores, geometry=[Point(r.lon, r.lat) for _, r in stores.iterrows()],
    crs="EPSG:4326"
)

# ---------------------------------------------------------------------------
# 3. Access score — gravity model + vehicle adjustment
# ---------------------------------------------------------------------------
def food_access(grid_lat, grid_lon, vehicle_rate):
    score = 0
    for _, store in stores_gdf.iterrows():
        dx = (grid_lon - store.lon) * 111.32 * np.cos(np.radians(grid_lat))
        dy = (grid_lat - store.lat) * 111.32
        dist_km = max(0.05, np.sqrt(dx**2 + dy**2))
        # USDA: within 1km walking or 10km driving
        walk_ok = 1.0 if dist_km <= 1.0 else 0.0
        drive_ok = 1.0 if dist_km <= 10.0 else 0.0
        effective_reach = vehicle_rate * drive_ok + (1 - vehicle_rate) * walk_ok
        weight = {"Supermarket": 3.0, "Market": 2.0, "Mini-Mart": 1.0}[store.type]
        score += (store.fresh_score * weight * effective_reach) / max(0.1, dist_km)
    return score

grid["food_access"] = grid.apply(
    lambda x: food_access(x["lat"], x["lon"], x["vehicle_rate"]), axis=1
)
grid["food_access_norm"] = (
    (grid["food_access"] - grid["food_access"].min()) /
    (grid["food_access"].max() - grid["food_access"].min())
) * 100

# Food desert = low access + low income
grid["is_food_desert"] = (grid["food_access_norm"] < 25) & (grid["income_index"] < 40)

def classify_food(row):
    if row["is_food_desert"]: return "Food Desert"
    elif row["food_access_norm"] < 35: return "Food Swamp"    # fast food only
    elif row["food_access_norm"] < 55: return "Low Access"
    elif row["food_access_norm"] < 75: return "Moderate Access"
    else: return "Good Access"

grid["food_class"] = grid.apply(classify_food, axis=1)

# ---------------------------------------------------------------------------
# 4. DBSCAN clustering to find contiguous food desert zones
# ---------------------------------------------------------------------------
desert_pts = grid[grid["is_food_desert"]][["lat", "lon"]].values
if len(desert_pts) > 0:
    coords_rad = np.radians(desert_pts)
    db = DBSCAN(eps=0.015, min_samples=3, metric="haversine").fit(coords_rad)
    grid.loc[grid["is_food_desert"], "desert_cluster"] = db.labels_
else:
    grid["desert_cluster"] = -1

# ---------------------------------------------------------------------------
# 5. Static map
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("#0a0a0a")
for ax in axes:
    ax.set_facecolor("#0a0a0a")

# Income index
norm_i = mcolors.Normalize(0, 100)
grid.plot(column="income_index", cmap="plasma", norm=norm_i, ax=axes[0], linewidth=0)
sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_i)
plt.colorbar(sm, ax=axes[0], label="Income Index (0–100)", shrink=0.8)
for _, store in stores_gdf.iterrows():
    mk = {"Supermarket": ("^", 120, "cyan"), "Market": ("s", 80, "lime"), "Mini-Mart": ("o", 50, "yellow")}
    m_shape, m_size, m_color = mk[store.type]
    axes[0].scatter(store.lon, store.lat, marker=m_shape, s=m_size,
                    color=m_color, zorder=5, edgecolors="white", linewidths=0.5)
axes[0].set_title("Income Distribution + Food Stores", color="white", fontsize=11, fontweight="bold")
axes[0].tick_params(colors="white")

# Food access score
norm_a = mcolors.Normalize(0, 100)
grid.plot(column="food_access_norm", cmap="RdYlGn", norm=norm_a, ax=axes[1], linewidth=0)
sm2 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm_a)
plt.colorbar(sm2, ax=axes[1], label="Food Access Score (0–100)", shrink=0.8)
axes[1].set_title("Food Access Score", color="white", fontsize=11, fontweight="bold")
axes[1].tick_params(colors="white")

# Food desert classification
class_colors = {
    "Food Desert": "#880e4f", "Food Swamp": "#d32f2f",
    "Low Access": "#f57c00", "Moderate Access": "#fbc02d", "Good Access": "#2e7d32"
}
grid["color"] = grid["food_class"].map(class_colors)
grid.plot(color=grid["color"], ax=axes[2], linewidth=0)
legend_elements = [Patch(facecolor=v, label=k) for k, v in class_colors.items()]
axes[2].legend(handles=legend_elements, loc="upper right", framealpha=0.85,
               facecolor="#111", labelcolor="white", fontsize=8)
axes[2].set_title("Food Access Classification", color="white", fontsize=11, fontweight="bold")
axes[2].tick_params(colors="white")

plt.suptitle("Food Desert Identification — Accra, Ghana",
             color="white", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/food_desert_map.png", dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print("Saved: outputs/food_desert_map.png")

# ---------------------------------------------------------------------------
# 6. Intervention gap chart
# ---------------------------------------------------------------------------
by_class = grid.groupby("food_class")["population"].sum()
order = ["Food Desert", "Food Swamp", "Low Access", "Moderate Access", "Good Access"]
by_class = by_class.reindex(order).fillna(0)

fig2, ax2 = plt.subplots(figsize=(11, 5))
fig2.patch.set_facecolor("#0a0a0a")
ax2.set_facecolor("#111")
colors_list = [class_colors[c] for c in order]
bars = ax2.bar(order, by_class.values / 1000, color=colors_list, edgecolor="none")
ax2.set_ylabel("Population (thousands)", color="white")
ax2.set_title("Population in Each Food Access Category", color="white", fontsize=13, fontweight="bold")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("#444")
for bar, val in zip(bars, by_class.values / 1000):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.0f}k", ha="center", color="white", fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/food_desert_population.png", dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print("Saved: outputs/food_desert_population.png")

# ---------------------------------------------------------------------------
# 7. Interactive map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.035, CITY_LON + 0.035], zoom_start=11,
               tiles="CartoDB dark_matter")

for _, row in grid[grid["food_class"] == "Food Desert"].iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=7,
        color="#880e4f", fill=True, fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>FOOD DESERT</b><br>Pop: {row['population']:,}<br>"
            f"Income Index: {row['income_index']:.0f}<br>"
            f"Access Score: {row['food_access_norm']:.1f}/100<br>"
            f"Vehicle Rate: {100*row['vehicle_rate']:.0f}%",
            max_width=200
        )
    ).add_to(m)

type_colors = {"Supermarket": "blue", "Market": "green", "Mini-Mart": "orange"}
for _, store in stores_gdf.iterrows():
    folium.CircleMarker(
        location=[store.lat, store.lon], radius=8,
        color=type_colors[store.type], fill=True, fill_opacity=0.9,
        popup=folium.Popup(
            f"<b>{store['name']}</b><br>Type: {store.type}<br>"
            f"Fresh Score: {store.fresh_score}/100", max_width=180
        )
    ).add_to(m)

title_html = """
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.85);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:15px;font-weight:bold;z-index:9999;">
Food Desert Identification — Accra, Ghana
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("outputs/food_desert_interactive.html")
print("Saved: outputs/food_desert_interactive.html")

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------
desert_pop = grid[grid["food_class"] == "Food Desert"]["population"].sum()
total_pop = grid["population"].sum()
print("\n" + "="*60)
print("  FOOD DESERT ANALYSIS — ACCRA, GHANA")
print("="*60)
print(f"  Total population         : {total_pop:,.0f}")
print(f"  Food desert population   : {desert_pop:,.0f} ({100*desert_pop/total_pop:.1f}%)")
print(f"  Food stores mapped       : {len(stores_gdf)}")
print(f"    Supermarkets           : {len(stores[stores.type=='Supermarket'])}")
print(f"    Markets                : {len(stores[stores.type=='Market'])}")
print(f"    Mini-Marts             : {len(stores[stores.type=='Mini-Mart'])}")
print(f"  Desert clusters (DBSCAN) : {int(grid['desert_cluster'].max()) + 1}")
print("\n  INTERVENTIONS NEEDED:")
print("  1. Mobile fresh-food vendors in food desert clusters")
print("  2. Community garden programs in low-income peripheral zones")
print("  3. Subsidised transport to existing markets")
print("  4. New market/cooperative in top 3 desert clusters")
print("="*60)
grid.drop(columns=["geometry"]).to_csv("outputs/food_desert_data.csv", index=False)
