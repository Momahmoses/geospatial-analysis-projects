# Project 04 — Food Desert Identification

## Overview
Identifies communities with poor access to fresh, affordable food by combining grocery store locations, income levels, vehicle ownership rates, and walking/driving distances. Uses USDA-adapted criteria and DBSCAN clustering to delineate contiguous food desert zones.

## Food Desert Criteria
A zone is classified as a food desert when **both** conditions are met:
- Food access score < 25th percentile
- Income index < 40/100 (low-income community)

## Methodology
1. **Population & income grid** — 45×45 cell grid with realistic spatial gradients
2. **Food outlet inventory** — Supermarkets, Markets, Mini-Marts with fresh-food scores
3. **Gravity access model** — `score = Σ (fresh_score × type_weight × reach) / distance`
4. **Vehicle-adjusted reach** — walking radius (1km) vs driving radius (10km) weighted by car ownership rate
5. **Classification** — Food Desert / Food Swamp / Low / Moderate / Good Access
6. **DBSCAN clustering** — identifies contiguous desert zones for intervention targeting

## Outputs
| File | Description |
|------|-------------|
| `outputs/food_desert_map.png` | Income, access score, and classification maps |
| `outputs/food_desert_population.png` | Population chart by access category |
| `outputs/food_desert_interactive.html` | Interactive map with store and desert details |
| `outputs/food_desert_data.csv` | Full per-cell dataset |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **USDA Food Access Research Atlas** — US food desert data
- **OpenStreetMap** — grocery/market/supermarket tags
- **WorldPop** — population grids
- **DHS household surveys** — vehicle ownership rates
- **World Bank PovcalNet** — income quintile data

## Impact
Food desert mapping helped Chicago, IL reduce food-insecure households by 18% through targeted mobile market routes and community garden investments in identified desert clusters.
