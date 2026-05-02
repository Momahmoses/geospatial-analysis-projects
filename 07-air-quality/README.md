# Project 07 — Air Quality & Pollution Dispersion Mapping

## Overview
Models PM2.5 and NO2 pollution dispersion from identified emission sources using a Gaussian plume model with wind direction and speed. Calculates population exposure, WHO guideline exceedances, and health burden per zone.

## Methodology
1. **Terrain & population grid** — 50×50 cells with elevation and population data
2. **Emission source inventory** — 6 sources: industrial zones, markets, traffic hubs, quarry
3. **Gaussian plume dispersion** — wind-driven concentration field for PM2.5 and NO2
4. **AQI classification** — Good / Moderate / Unhealthy for Sensitive / Unhealthy / Very Unhealthy / Hazardous
5. **Population exposure** — WHO threshold exceedances × population count
6. **Health burden index** — population-weighted pollution load per zone

## Key Metrics
- PM2.5 WHO annual guideline: **5 µg/m³** (target) / **15 µg/m³** (interim)
- NO2 WHO annual guideline: **10 µg/m³** (target) / **25 µg/m³** (interim)

## Outputs
| File | Description |
|------|-------------|
| `outputs/air_quality_map.png` | PM2.5, NO2, AQI, and health burden maps |
| `outputs/air_quality_sources.png` | Emission source comparison chart |
| `outputs/air_quality_interactive.html` | Interactive PM2.5 heatmap with source markers |
| `outputs/air_quality_data.csv` | Per-cell concentration and AQI data |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **OpenAQ** — open air quality sensor network
- **Copernicus Atmosphere Monitoring Service (CAMS)** — European air quality data
- **NASA MERRA-2** — global atmospheric reanalysis
- **GEOS-Chem** — global atmospheric chemistry model
- **WHO Global Air Quality Database**

## Impact
Air quality mapping in Nairobi using similar methodology enabled targeted enforcement of emission regulations, reducing PM2.5 in industrial corridors by 28% over two years.
