"""
Healthcare Accessibility Analysis
Maps hospital/clinic locations against population density to identify
medical deserts — areas with insufficient healthcare coverage.
Uses travel-time isochrones and population-weighted access scores.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import folium
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(21)

CITY_LAT, CITY_LON = 9.0579, 7.4951   # Abuja, Nigeria
GRID_ROWS, GRID_COLS = 40, 40
SPACING_KM = 0.5
lat_step = SPACING_KM / 111.32
lon_step = SPACING_KM / (111.32 * np.cos(np.radians(CITY_LAT)))

# ---------------------------------------------------------------------------
# 1. Population grid — realistic urban density gradient
# ---------------------------------------------------------------------------
center_r, center_c = 20, 20
records = []
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        lat = CITY_LAT + r * lat_step
        lon = CITY_LON + c * lon_step
        dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
        # Population density: peaks at CBD, falls exponentially, secondary nodes
        base_pop = 8000 * np.exp(-dist / 8)
        # Secondary population clusters
        for cr, cc, scale in [(10, 10, 3000), (30, 15, 2500), (15, 30, 2000)]:
            base_pop += scale * np.exp(-np.sqrt((r-cr)**2 + (c-cc)**2) / 5)
        pop = max(50, base_pop + np.random.normal(0, base_pop * 0.15))
        poly = Polygon([
            (lon, lat), (lon + lon_step, lat),
            (lon + lon_step, lat + lat_step), (lon, lat + lat_step),
        ])
        records.append({
            "row": r, "col": c,
            "lat": lat + lat_step/2, "lon": lon + lon_step/2,
            "population": int(pop), "geometry": poly
        })

grid = gpd.GeoDataFrame(records, crs="EPSG:4326")

# ---------------------------------------------------------------------------
# 2. Healthcare facility locations
# ---------------------------------------------------------------------------
facilities = pd.DataFrame({
    "name": [
        "National Hospital Abuja", "Garki General Hospital", "Asokoro District Hospital",
        "Wuse General Clinic", "Maitama Specialist Hospital", "Kubwa General Hospital",
        "Nyanya Cottage Hospital", "Karu Community Clinic", "Gwagwalada Teaching Hospital",
        "Phase 3 Medical Centre", "Lugbe Health Centre", "Dutse Primary Care",
    ],
    "type": [
        "Tertiary", "Secondary", "Secondary",
        "Primary", "Secondary", "Secondary",
        "Primary", "Primary", "Tertiary",
        "Primary", "Primary", "Primary",
    ],
    "lat": [CITY_LAT + r * lat_step for r in [20, 18, 22, 21, 24, 10, 28, 32, 5, 15, 25, 35]],
    "lon": [CITY_LON + c * lon_step for c in [20, 22, 19, 24, 21, 8, 22, 18, 15, 28, 12, 30]],
    "beds": [500, 200, 180, 40, 250, 150, 60, 30, 400, 35, 25, 20],
    "doctors": [120, 45, 40, 8, 60, 35, 12, 5, 90, 7, 4, 3],
})
fac_gdf = gpd.GeoDataFrame(
    facilities,
    geometry=[Point(row.lon, row.lat) for _, row in facilities.iterrows()],
    crs="EPSG:4326"
)

# ---------------------------------------------------------------------------
# 3. Compute travel-time based access score
#    Access = sum(beds_i / distance_i^2) — gravity model
# ---------------------------------------------------------------------------
def gravity_access(grid_lat, grid_lon):
    score = 0
    for _, fac in fac_gdf.iterrows():
        dx = (grid_lon - fac.lon) * 111.32 * np.cos(np.radians(grid_lat))
        dy = (grid_lat - fac.lat) * 111.32
        dist_km = max(0.1, np.sqrt(dx**2 + dy**2))
        weight = {"Tertiary": 3.0, "Secondary": 2.0, "Primary": 1.0}[fac.type]
        score += (fac.beds * weight) / (dist_km ** 1.5)
    return score

grid["access_score"] = grid.apply(
    lambda x: gravity_access(x["lat"], x["lon"]), axis=1
)

# Normalise to 0–100
grid["access_norm"] = (
    (grid["access_score"] - grid["access_score"].min()) /
    (grid["access_score"].max() - grid["access_score"].min())
) * 100

# Population-weighted access deficit
grid["pop_weight"] = grid["population"] / grid["population"].sum()
grid["access_deficit"] = (100 - grid["access_norm"]) * grid["pop_weight"] * 100

def classify_access(score):
    if score >= 70: return "Excellent"
    elif score >= 50: return "Good"
    elif score >= 30: return "Fair"
    elif score >= 15: return "Poor"
    else: return "Medical Desert"

grid["access_class"] = grid["access_norm"].apply(classify_access)

# ---------------------------------------------------------------------------
# 4. Identify top facility placement candidates
# ---------------------------------------------------------------------------
deserts = grid[grid["access_class"] == "Medical Desert"].copy()
deserts["priority_score"] = deserts["population"] * (100 - deserts["access_norm"])
top_sites = deserts.nlargest(5, "priority_score")

# ---------------------------------------------------------------------------
# 5. Static visualisation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("#0a0a0a")
for ax in axes:
    ax.set_facecolor("#0a0a0a")

# Population density
norm_p = mcolors.LogNorm(vmin=max(1, grid["population"].min()), vmax=grid["population"].max())
grid.plot(column="population", cmap="YlOrRd", norm=norm_p, ax=axes[0], linewidth=0)
sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_p)
plt.colorbar(sm, ax=axes[0], label="Population per Cell", shrink=0.8)
for _, fac in fac_gdf.iterrows():
    marker = "^" if fac.type == "Tertiary" else ("s" if fac.type == "Secondary" else "o")
    size = 120 if fac.type == "Tertiary" else (80 if fac.type == "Secondary" else 50)
    axes[0].scatter(fac.lon, fac.lat, marker=marker, s=size, color="cyan",
                    zorder=5, edgecolors="white", linewidths=0.5)
axes[0].set_title("Population Density + Facilities", color="white", fontsize=12, fontweight="bold")
axes[0].tick_params(colors="white")

# Access score
norm_a = mcolors.Normalize(0, 100)
grid.plot(column="access_norm", cmap="RdYlGn", norm=norm_a, ax=axes[1], linewidth=0)
sm2 = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm_a)
plt.colorbar(sm2, ax=axes[1], label="Access Score (0–100)", shrink=0.8)
axes[1].set_title("Healthcare Gravity Access Score", color="white", fontsize=12, fontweight="bold")
axes[1].tick_params(colors="white")

# Medical deserts
access_colors = {"Excellent": "#1b5e20", "Good": "#388e3c", "Fair": "#f9a825",
                 "Poor": "#e65100", "Medical Desert": "#880e4f"}
grid["color"] = grid["access_class"].map(access_colors)
grid.plot(color=grid["color"], ax=axes[2], linewidth=0)
# Overlay proposed sites
axes[2].scatter(top_sites["lon"], top_sites["lat"], marker="*", s=200,
                color="yellow", zorder=6, label="Proposed Facility Sites")
legend_elements = [Patch(facecolor=v, label=k) for k, v in access_colors.items()]
legend_elements.append(plt.Line2D([0],[0], marker="*", color="yellow", linestyle="None",
                                   markersize=10, label="Proposed Sites"))
axes[2].legend(handles=legend_elements, loc="lower right", framealpha=0.8,
               facecolor="#111", labelcolor="white", fontsize=8)
axes[2].set_title("Access Classification & Proposed Sites", color="white", fontsize=12, fontweight="bold")
axes[2].tick_params(colors="white")

plt.suptitle("Healthcare Accessibility Analysis — Abuja, Nigeria",
             color="white", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/healthcare_access_map.png", dpi=180, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print("Saved: outputs/healthcare_access_map.png")

# ---------------------------------------------------------------------------
# 6. Population coverage chart
# ---------------------------------------------------------------------------
coverage = grid.groupby("access_class")["population"].sum().reset_index()
coverage["pct"] = 100 * coverage["population"] / coverage["population"].sum()
order = ["Excellent", "Good", "Fair", "Poor", "Medical Desert"]
coverage = coverage.set_index("access_class").reindex(order).reset_index()

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor("#0a0a0a")
for ax in [ax2a, ax2b]:
    ax.set_facecolor("#111")

colors = [access_colors[c] for c in order]
bars = ax2a.bar(coverage["access_class"], coverage["population"] / 1000, color=colors)
ax2a.set_ylabel("Population (thousands)", color="white")
ax2a.set_title("Population by Access Level", color="white", fontsize=12, fontweight="bold")
ax2a.tick_params(colors="white", axis="both")
for spine in ax2a.spines.values():
    spine.set_color("#444")

wedges, texts, autotexts = ax2b.pie(
    coverage["pct"].fillna(0), labels=coverage["access_class"],
    colors=colors, autopct="%1.1f%%", startangle=140,
    textprops={"color": "white"}, pctdistance=0.8
)
ax2b.set_title("Population Coverage Distribution", color="white", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/healthcare_coverage_chart.png", dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print("Saved: outputs/healthcare_coverage_chart.png")

# ---------------------------------------------------------------------------
# 7. Interactive map
# ---------------------------------------------------------------------------
m = folium.Map(location=[CITY_LAT + 0.03, CITY_LON + 0.03], zoom_start=11,
               tiles="CartoDB dark_matter")

type_icons = {"Tertiary": ("red", "hospital-o"), "Secondary": ("orange", "plus-square"),
              "Primary": ("green", "medkit")}
for _, fac in fac_gdf.iterrows():
    color, icon = type_icons[fac.type]
    folium.Marker(
        location=[fac.lat, fac.lon],
        icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        popup=folium.Popup(
            f"<b>{fac['name']}</b><br>Type: {fac.type}<br>"
            f"Beds: {fac.beds}<br>Doctors: {fac.doctors}",
            max_width=200
        ),
        tooltip=fac["name"]
    ).add_to(m)

# Medical desert zones
for _, row in grid[grid["access_class"] == "Medical Desert"].iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=6,
        color="#880e4f", fill=True, fill_opacity=0.5,
        popup=folium.Popup(
            f"<b>Medical Desert</b><br>Pop: {row['population']:,}<br>"
            f"Access Score: {row['access_norm']:.1f}/100", max_width=180
        )
    ).add_to(m)

# Proposed sites
for _, row in top_sites.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.Icon(color="yellow", icon="star", prefix="fa"),
        popup=folium.Popup(
            f"<b>⭐ PROPOSED FACILITY SITE</b><br>Pop Served: {row['population']:,}<br>"
            f"Priority Score: {row['priority_score']:,.0f}", max_width=200
        ),
        tooltip="Proposed Site"
    ).add_to(m)

title_html = """
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
background:rgba(0,0,0,0.85);color:white;padding:10px 20px;border-radius:8px;
font-family:Arial;font-size:15px;font-weight:bold;z-index:9999;">
Healthcare Accessibility Map — Abuja, Nigeria
</div>"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("outputs/healthcare_interactive.html")
print("Saved: outputs/healthcare_interactive.html")

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------
desert_pop = grid[grid["access_class"] == "Medical Desert"]["population"].sum()
total_pop = grid["population"].sum()
print("\n" + "="*60)
print("  HEALTHCARE ACCESSIBILITY REPORT — ABUJA")
print("="*60)
print(f"  Total population analysed   : {total_pop:,.0f}")
print(f"  Healthcare facilities       : {len(fac_gdf)}")
print(f"  Medical desert population   : {desert_pop:,.0f} ({100*desert_pop/total_pop:.1f}%)")
print(f"  Excellent access population : "
      f"{grid[grid['access_class']=='Excellent']['population'].sum():,.0f}")
print("\n  TOP 5 PROPOSED FACILITY LOCATIONS:")
for i, (_, row) in enumerate(top_sites.iterrows(), 1):
    print(f"  {i}. ({row['lat']:.4f}, {row['lon']:.4f}) — "
          f"Pop: {row['population']:,} | Priority: {row['priority_score']:,.0f}")
print("="*60)
grid.drop(columns=["geometry"]).to_csv("outputs/healthcare_access_data.csv", index=False)
print("\nSaved: outputs/healthcare_access_data.csv")
