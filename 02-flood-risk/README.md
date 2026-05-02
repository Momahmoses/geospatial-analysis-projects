# Project 02 — Flood Risk & Early Warning System

## Overview
Multi-factor flood risk mapping combining terrain elevation, rainfall intensity, river proximity, and soil drainage capacity. Produces risk scores per zone and triggers automated early warning alerts.

## Methodology
1. **Digital Elevation Model** — smooth terrain with river valley simulation
2. **River network** — main river + tributaries, proximity scoring
3. **Rainfall intensity** — spatial storm simulation (mm/hr)
4. **Soil drainage** — beta-distributed drainage capacity, reduced in urban areas
5. **Composite risk score** — weighted combination of 4 factors (0–100)
6. **Early warning alerts** — automatic flagging of Extreme/High zones

## Risk Factor Weights
| Factor | Weight | Rationale |
|--------|--------|-----------|
| Elevation | 35% | Low-lying areas flood first |
| Rainfall | 25% | Storm intensity drives acute events |
| River proximity | 25% | Overflow and backwater flooding |
| Soil drainage | 15% | Impervious surfaces worsen runoff |

## Outputs
| File | Description |
|------|-------------|
| `outputs/flood_risk_map.png` | 3-panel: Elevation, Rainfall, Risk zones |
| `outputs/flood_risk_distribution.png` | Risk class distribution bar chart |
| `outputs/flood_risk_interactive.html` | Interactive warning map with popups |
| `outputs/flood_risk_data.csv` | Full dataset per grid cell |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **SRTM / ASTER DEM** — 30m elevation (NASA/USGS)
- **GPM IMERG** — Global precipitation measurement
- **OpenStreetMap** — River and waterway features
- **HydroSHEDS** — Hydrological drainage basins
- **FAO HWSD** — Soil drainage classification

## Impact
Early warning systems based on this approach have been shown to reduce flood fatalities by up to 35% in developing countries by giving communities 6–24 hours of evacuation lead time.
