"""
Crop Yield & Agricultural Optimisation
Uses multi-band satellite-derived indices (NDVI, NDWI, soil moisture proxy)
to identify underperforming farmland, optimize irrigation zones, and
forecast yield gaps. Covers smallholder farming landscape.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
np.random.seed(101)

REGION = "Savannah Agricultural Belt — Northern Ghana"
ROWS, COLS = 60, 60
CELL_HA = 100   # 1km grid cells (100 ha each)

# ---------------------------------------------------------------------------
# 1. Biophysical landscape simulation
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import gaussian_filter
    soil_noise = gaussian_filter(np.random.rand(ROWS, COLS), sigma=7)
    rain_noise = gaussian_filter(np.random.rand(ROWS, COLS), sigma=10)
    slope_noise = gaussian_filter(np.random.rand(ROWS, COLS), sigma=4)
except ImportError:
    soil_noise = np.random.rand(ROWS, COLS)
    rain_noise = np.random.rand(ROWS, COLS)
    slope_noise = np.random.rand(ROWS, COLS)

# Soil fertility index (0–1): higher in valley bottoms
soil_fertility = np.zeros((ROWS, COLS))
for r in range(ROWS):
    dist_valley = abs(r - ROWS // 2) / (ROWS // 2)
    for c in range(COLS):
        soil_fertility[r, c] = max(0.1, 0.8 - 0.6 * dist_valley + 0.3 * soil_noise[r, c])

# Rainfall accumulation (mm): gradient from south to north
rainfall_mm = np.zeros((ROWS, COLS))
for r in range(ROWS):
    base = 1100 - 6 * r   # 1100mm south to 740mm north
    for c in range(COLS):
        rainfall_mm[r, c] = max(400, base + 100 * rain_noise[r, c] - 50)

# Slope (degrees): affects erosion and runoff
slope_deg = 2 + 18 * slope_noise

# Irrigation access (binary + partial)
irrigation = np.zeros((ROWS, COLS))
# Two irrigation canals running east-west
for r in [15, 40]:
    for c in range(COLS):
        dist = abs(r - 15) if r == 15 else abs(r - 40)
        irrigation[r, c] = max(0, 1 - dist / 8)
    # Extend access to nearby cells
    for dr in range(-7, 8):
        for c in range(COLS):
            nr = r + dr
            if 0 <= nr < ROWS:
                irrigation[nr, c] = max(irrigation[nr, c], max(0, 1 - abs(dr)/7))

# ---------------------------------------------------------------------------
# 2. Crop type assignment (rainfed vs irrigated zones)
# ---------------------------------------------------------------------------
crops = np.empty((ROWS, COLS), dtype=object)
for r in range(ROWS):
    for c in range(COLS):
        if irrigation[r, c] > 0.5:
            crop = np.random.choice(["Rice", "Tomato", "Onion"], p=[0.5, 0.3, 0.2])
        elif soil_fertility[r, c] > 0.6:
            crop = np.random.choice(["Maize", "Sorghum", "Groundnut"], p=[0.5, 0.3, 0.2])
        elif rainfall_mm[r, c] > 900:
            crop = np.random.choice(["Maize", "Cassava"], p=[0.6, 0.4])
        else:
            crop = np.random.choice(["Sorghum", "Millet", "Groundnut"], p=[0.4, 0.35, 0.25])
        crops[r, c] = crop

# ---------------------------------------------------------------------------
# 3. NDVI time series (growing season: Apr–Sep)
# ---------------------------------------------------------------------------
MONTHS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
ndvi_series = {}
for month_idx, month in enumerate(MONTHS):
    ndvi = np.zeros((ROWS, COLS))
    for r in range(ROWS):
        for c in range(COLS):
            base_ndvi = (
                0.3 * soil_fertility[r, c] +
                0.25 * (rainfall_mm[r, c] - 400) / 700 +
                0.25 * irrigation[r, c] +
                0.1 * (1 - slope_deg[r, c] / 20) +
                0.1 * np.random.rand()
            )
            # Seasonal curve: peaks in July-August
            season_curve = np.sin(np.pi * (month_idx + 1) / (len(MONTHS) + 1))
            ndvi[r, c] = max(0.1, min(0.95, base_ndvi * season_curve + 0.1))
    ndvi_series[month] = ndvi

peak_ndvi = ndvi_series["Aug"]

# ---------------------------------------------------------------------------
# 4. Yield estimation (t/ha) — empirical relationships
# ---------------------------------------------------------------------------
YIELD_PARAMS = {
    "Maize":    {"max_yield": 8.0, "ndvi_coef": 6.0, "rain_coef": 0.003},
    "Rice":     {"max_yield": 6.0, "ndvi_coef": 5.0, "rain_coef": 0.002},
    "Sorghum":  {"max_yield": 4.5, "ndvi_coef": 4.0, "rain_coef": 0.001},
    "Millet":   {"max_yield": 3.5, "ndvi_coef": 3.5, "rain_coef": 0.0015},
    "Cassava":  {"max_yield": 12.0, "ndvi_coef": 8.0, "rain_coef": 0.002},
    "Groundnut":{"max_yield": 3.0, "ndvi_coef": 2.5, "rain_coef": 0.001},
    "Tomato":   {"max_yield": 30.0, "ndvi_coef": 20.0, "rain_coef": 0.005},
    "Onion":    {"max_yield": 20.0, "ndvi_coef": 15.0, "rain_coef": 0.003},
}

actual_yield = np.zeros((ROWS, COLS))
potential_yield = np.zeros((ROWS, COLS))
yield_gap = np.zeros((ROWS, COLS))
yield_gap_pct = np.zeros((ROWS, COLS))

for r in range(ROWS):
    for c in range(COLS):
        crop = crops[r, c]
        params = YIELD_PARAMS[crop]
        act = (params["ndvi_coef"] * peak_ndvi[r, c] +
               params["rain_coef"] * rainfall_mm[r, c]) * soil_fertility[r, c]
        act = min(params["max_yield"] * 0.95, max(0.1, act + np.random.normal(0, 0.2)))
        pot = params["max_yield"] * soil_fertility[r, c]
        actual_yield[r, c] = round(act, 2)
        potential_yield[r, c] = round(pot, 2)
        yield_gap[r, c] = max(0, pot - act)
        yield_gap_pct[r, c] = 100 * yield_gap[r, c] / pot if pot > 0 else 0

# ---------------------------------------------------------------------------
# 5. Irrigation optimisation — identify cells with high yield gap but good soil
# ---------------------------------------------------------------------------
priority_irrigation = (yield_gap_pct > 40) & (soil_fertility > 0.5) & (irrigation < 0.3)

# Fertiliser need
fertiliser_need = np.where(
    (soil_fertility < 0.35) & (yield_gap_pct > 30), "High",
    np.where(
        (soil_fertility < 0.55) & (yield_gap_pct > 20), "Moderate",
        "Low"
    )
)

# ---------------------------------------------------------------------------
# 6. Static visualisation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(21, 14))
fig.patch.set_facecolor("#0a0f0a")
for ax in axes.flat:
    ax.set_facecolor("#0a0f0a")

# Soil fertility
norm_sf = mcolors.Normalize(0, 1)
im0 = axes[0][0].imshow(soil_fertility, cmap="YlOrBr", norm=norm_sf, origin="lower")
plt.colorbar(im0, ax=axes[0][0], label="Soil Fertility Index", shrink=0.8)
axes[0][0].set_title("Soil Fertility Index", color="white", fontsize=11, fontweight="bold")
axes[0][0].tick_params(colors="white")

# Rainfall
norm_rain = mcolors.Normalize(400, 1200)
im1 = axes[0][1].imshow(rainfall_mm, cmap="Blues", norm=norm_rain, origin="lower")
plt.colorbar(im1, ax=axes[0][1], label="Rainfall (mm)", shrink=0.8)
axes[0][1].set_title("Seasonal Rainfall (mm)", color="white", fontsize=11, fontweight="bold")
axes[0][1].tick_params(colors="white")

# Peak NDVI
norm_ndvi = mcolors.Normalize(0.1, 0.95)
im2 = axes[0][2].imshow(peak_ndvi, cmap="RdYlGn", norm=norm_ndvi, origin="lower")
plt.colorbar(im2, ax=axes[0][2], label="NDVI (Aug peak)", shrink=0.8)
axes[0][2].set_title("Peak NDVI (August)", color="white", fontsize=11, fontweight="bold")
axes[0][2].tick_params(colors="white")

# Actual yield
norm_ay = mcolors.Normalize(0, actual_yield.max())
im3 = axes[1][0].imshow(actual_yield, cmap="Greens", norm=norm_ay, origin="lower")
plt.colorbar(im3, ax=axes[1][0], label="Actual Yield (t/ha)", shrink=0.8)
axes[1][0].set_title("Actual Crop Yield (t/ha)", color="white", fontsize=11, fontweight="bold")
axes[1][0].tick_params(colors="white")

# Yield gap
norm_yg = mcolors.Normalize(0, yield_gap_pct.max())
im4 = axes[1][1].imshow(yield_gap_pct, cmap="RdYlGn_r", norm=norm_yg, origin="lower")
plt.colorbar(im4, ax=axes[1][1], label="Yield Gap (%)", shrink=0.8)
axes[1][1].set_title("Yield Gap (% of Potential)", color="white", fontsize=11, fontweight="bold")
axes[1][1].tick_params(colors="white")

# Irrigation priority
axes[1][2].imshow(priority_irrigation.astype(int), cmap="RdYlGn", origin="lower")
axes[1][2].imshow(irrigation, cmap="Blues", alpha=0.4, origin="lower")
legend_elements = [
    Patch(facecolor="#c62828", label="Priority Irrigation Zone"),
    Patch(facecolor="#1565c0", label="Existing Irrigation"),
]
axes[1][2].legend(handles=legend_elements, loc="upper right", framealpha=0.85,
                  facecolor="#111", labelcolor="white", fontsize=8)
axes[1][2].set_title("Irrigation Priority Zones", color="white", fontsize=11, fontweight="bold")
axes[1][2].tick_params(colors="white")

plt.suptitle(f"Crop Yield Optimisation — {REGION}",
             color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/crop_yield_map.png", dpi=180, bbox_inches="tight", facecolor="#0a0f0a")
plt.close()
print("Saved: outputs/crop_yield_map.png")

# ---------------------------------------------------------------------------
# 7. NDVI seasonal profile chart
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
fig2.patch.set_facecolor("#0a0f0a")
for ax in axes2:
    ax.set_facecolor("#111")

crops_list = list(YIELD_PARAMS.keys())
colors_crop = plt.cm.tab10(np.linspace(0, 1, len(crops_list)))

# Average NDVI per month
for crop, color in zip(crops_list, colors_crop):
    mask = crops == crop
    if mask.sum() == 0:
        continue
    monthly_ndvi = [ndvi_series[m][mask].mean() for m in MONTHS]
    axes2[0].plot(MONTHS, monthly_ndvi, marker="o", label=crop, color=color, linewidth=2)

axes2[0].set_ylabel("Mean NDVI", color="white")
axes2[0].set_title("Seasonal NDVI Profile by Crop Type", color="white", fontsize=12, fontweight="bold")
axes2[0].legend(framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=8, ncol=2)
axes2[0].tick_params(colors="white")
for spine in axes2[0].spines.values():
    spine.set_color("#444")

# Yield gap by crop
crop_gaps = []
for crop in crops_list:
    mask = crops == crop
    if mask.sum() == 0:
        continue
    avg_gap = yield_gap_pct[mask].mean()
    avg_actual = actual_yield[mask].mean()
    avg_potential = potential_yield[mask].mean()
    crop_gaps.append({"crop": crop, "actual": avg_actual, "gap": avg_potential - avg_actual})

df_gaps = pd.DataFrame(crop_gaps).sort_values("gap", ascending=True)
bars_act = axes2[1].barh(df_gaps["crop"], df_gaps["actual"], color="#2e7d32", label="Actual Yield")
bars_gap = axes2[1].barh(df_gaps["crop"], df_gaps["gap"],
                          left=df_gaps["actual"], color="#c62828", alpha=0.7, label="Yield Gap")
axes2[1].set_xlabel("Yield (t/ha)", color="white")
axes2[1].set_title("Actual Yield vs Yield Gap by Crop", color="white", fontsize=12, fontweight="bold")
axes2[1].tick_params(colors="white")
axes2[1].legend(framealpha=0.8, facecolor="#111", labelcolor="white", fontsize=9)
for spine in axes2[1].spines.values():
    spine.set_color("#444")

plt.tight_layout()
plt.savefig("outputs/crop_yield_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0a0f0a")
plt.close()
print("Saved: outputs/crop_yield_analysis.png")

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------
total_ha = ROWS * COLS * CELL_HA
priority_ha = priority_irrigation.sum() * CELL_HA
total_actual = actual_yield.mean() * total_ha / 1000   # tonnes
total_potential = potential_yield.mean() * total_ha / 1000
yield_gap_total = total_potential - total_actual

print("\n" + "="*60)
print(f"  CROP YIELD ANALYSIS — {REGION}")
print("="*60)
print(f"  Total area analysed         : {total_ha:,} ha")
print(f"  Mean actual yield           : {actual_yield.mean():.2f} t/ha")
print(f"  Mean potential yield        : {potential_yield.mean():.2f} t/ha")
print(f"  Average yield gap           : {yield_gap_pct.mean():.1f}%")
print(f"  Total production gap        : {yield_gap_total:,.0f} tonnes/season")
print(f"  Priority irrigation areas   : {priority_ha:,} ha")
print(f"  High fertiliser need zones  : {(fertiliser_need=='High').sum():,} cells")
print("\n  TOP INTERVENTIONS:")
print(f"  1. Expand drip irrigation to {priority_ha:,.0f} ha priority zones")
print("  2. Fertiliser subsidy for high-need smallholders")
print("  3. Crop variety improvement (drought-tolerant maize/sorghum)")
print("  4. Early warning for rainfall deficit years")
print("  5. Soil health restoration in low-fertility belts")
print("="*60)

crop_summary = []
for crop in crops_list:
    mask = crops == crop
    if mask.sum() == 0:
        continue
    crop_summary.append({
        "crop": crop, "cells": mask.sum(), "area_ha": mask.sum() * CELL_HA,
        "mean_actual_yield": round(actual_yield[mask].mean(), 2),
        "mean_potential_yield": round(potential_yield[mask].mean(), 2),
        "mean_yield_gap_pct": round(yield_gap_pct[mask].mean(), 1),
    })
pd.DataFrame(crop_summary).to_csv("outputs/crop_yield_summary.csv", index=False)
print("\nSaved: outputs/crop_yield_summary.csv")
