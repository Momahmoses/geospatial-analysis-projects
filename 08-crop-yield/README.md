# Project 08 — Crop Yield & Agricultural Optimisation

## Overview
Uses satellite-derived vegetation indices (NDVI, NDWI), soil data, rainfall gradients, and irrigation access to estimate actual crop yields, identify yield gaps, and prioritise areas for irrigation expansion and fertiliser intervention.

## Methodology
1. **Biophysical landscape** — Soil fertility, rainfall gradient, slope, irrigation access
2. **Crop type mapping** — 8 crop types assigned by growing conditions
3. **NDVI time series** — 6-month growing season (April–September)
4. **Yield estimation** — Empirical NDVI × soil fertility × rainfall model
5. **Yield gap analysis** — Actual vs potential yield by crop and zone
6. **Irrigation priority scoring** — High yield gap + good soil + no current irrigation

## Crops Modelled
Maize, Rice, Sorghum, Millet, Cassava, Groundnut, Tomato, Onion

## Outputs
| File | Description |
|------|-------------|
| `outputs/crop_yield_map.png` | 6-panel: soil, rainfall, NDVI, yield, gap, irrigation priority |
| `outputs/crop_yield_analysis.png` | Seasonal NDVI profiles + yield gap by crop |
| `outputs/crop_yield_summary.csv` | Per-crop yield statistics |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **Sentinel-2 / Landsat** — NDVI, NDWI time series
- **ISRIC World Soil Database** — soil fertility and texture
- **CHIRPS / TAMSAT** — rainfall estimates for Africa
- **FAO AQUASTAT** — irrigation coverage
- **GYGA (Global Yield Gap Atlas)** — benchmark yield data by crop/region

## Impact
Yield gap analysis guided by satellite data has helped the Alliance of Bioversity International target extension services in Ghana, reducing average maize yield gaps by 30% in 3 seasons.
