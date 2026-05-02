# Project 01 — Urban Heat Island Mapping

## Overview
Identifies city zones with dangerous heat buildup by simulating Land Surface Temperature (LST) across land-use types. Outputs priority zones for tree planting and cool-roof policy interventions.

## Methodology
1. **City grid generation** — 50×50 grid of 200m cells covering 100 km²
2. **Land-use classification** — 7 categories (CBD, Residential, Industrial, Green Space, etc.)
3. **LST simulation** — temperature modeled from land-use type + impervious surface fraction + noise
4. **NDVI calculation** — vegetation index per zone
5. **UHI intensity** — delta from rural baseline temperature
6. **Risk classification** — Critical / High / Moderate / Low / Cool Zone

## Key Outputs
| File | Description |
|------|-------------|
| `outputs/uhi_static_map.png` | 3-panel map: LST, NDVI, Risk zones |
| `outputs/uhi_landuse_chart.png` | Mean LST bar chart by land-use |
| `outputs/uhi_interactive_map.html` | Interactive heatmap with popups |
| `outputs/uhi_summary_stats.csv` | Statistics per land-use type |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **NASA MODIS LST** — `MOD11A2` product (1km, 8-day composites)
- **Landsat 8/9 Band 10** — 100m thermal infrared
- **OpenStreetMap** — land-use polygons via `osmnx`
- **Copernicus Urban Atlas** — EU cities

## Impact
Enables city planners to target cooling interventions (trees, reflective surfaces, parks) precisely where surface temperatures are highest — reducing heat-related mortality and energy consumption.
