# Project 05 — Deforestation & Land Use Change Monitoring

## Overview
Tracks forest cover loss over a 10-year period (2015–2024) using simulated NDVI raster time series. Detects deforestation frontiers, quantifies annual and cumulative loss, and identifies spatial drivers (agriculture, roads, mining).

## Methodology
1. **Baseline forest map (2015)** — NDVI surface with realistic forest gradient
2. **Annual simulation** — deforestation pressure from 3 spatial sources
3. **Change detection** — pixel-level forest/non-forest classification per year
4. **Driver attribution** — spatial clustering of loss by direction/source
5. **Trend analysis** — 10-year time series of cover, loss rate, and NDVI health

## Deforestation Drivers Modelled
| Driver | Direction | Relative Contribution |
|--------|-----------|----------------------|
| Agricultural expansion | East | ~45% |
| Road/urban sprawl | South-West | ~32% |
| Artisanal mining | North | ~23% |

## Outputs
| File | Description |
|------|-------------|
| `outputs/deforestation_maps.png` | NDVI snapshots + change detection map + annual loss bar |
| `outputs/deforestation_trends.png` | 3-panel: cover, loss rate, NDVI trend |
| `outputs/deforestation_stats.csv` | Year-by-year forest statistics |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **Global Forest Watch** — Hansen/UMD tree cover loss (30m, annual)
- **Landsat 8/9** — NDVI time series
- **Sentinel-2** — 10m multispectral imagery
- **PRODES** — Amazon deforestation monitoring (Brazil)
- **FIRMS** — NASA fire detection (active clearing proxy)

## Impact
Forest monitoring systems like this underpin REDD+ carbon credit programmes and have been used to halt illegal clearing in protected areas across the Congo Basin, Amazon, and Borneo.
