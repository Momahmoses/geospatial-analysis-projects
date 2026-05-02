# Project 03 — Healthcare Accessibility Analysis

## Overview
Maps hospital and clinic coverage against population distribution to identify **medical deserts** — areas where populations lack adequate access to healthcare. Uses a gravity model to compute continuous access scores, then proposes optimal locations for new facilities.

## Methodology
1. **Population grid** — synthetic density surface with urban clusters
2. **Facility inventory** — 12 facilities (Tertiary/Secondary/Primary) with beds & doctors
3. **Gravity access model** — `score = Σ (beds × type_weight) / distance^1.5`
4. **Access classification** — Excellent / Good / Fair / Poor / Medical Desert
5. **Site optimisation** — highest priority = largest underserved population × lowest access

## Outputs
| File | Description |
|------|-------------|
| `outputs/healthcare_access_map.png` | Population, access score, and classification maps |
| `outputs/healthcare_coverage_chart.png` | Population distribution by access level |
| `outputs/healthcare_interactive.html` | Interactive map with facility details |
| `outputs/healthcare_access_data.csv` | Per-cell access scores and classifications |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **WHO Global Health Facility Registry** — facility locations
- **WorldPop** — 100m resolution population grids
- **OpenStreetMap** — hospital/clinic amenity tags
- **DHS Program** — healthcare utilisation surveys
- **OpenRouteService API** — real travel time isochrones

## Impact
This analysis approach was used by UNICEF Nigeria and PATH to prioritise primary healthcare centre (PHC) construction in underserved LGAs, improving coverage for ~2.3 million people.
