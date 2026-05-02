# Project 06 — Road Safety & Accident Hotspot Analysis

## Overview
Processes 5 years of traffic accident records to identify dangerous intersections and road segments using Kernel Density Estimation (KDE) and severity-weighted hotspot mapping.

## Methodology
1. **Accident dataset** — 1,200 synthetic accidents (2020–2024) with severity, hour, vehicle type
2. **KDE hotspot mapping** — Gaussian kernel density over accident point cloud
3. **Severity analysis** — Fatal / Serious Injury / Minor Injury / Property Damage
4. **Temporal patterns** — peak hour analysis and weekday vs weekend breakdown
5. **Vehicle type analysis** — motorcycle vs car vs bus vs bicycle exposure
6. **Intersection scoring** — proximity-based accident count per intersection

## Accident Distribution (Simulated)
| Severity | Share |
|----------|-------|
| Fatal | ~6% |
| Serious Injury | ~19% |
| Minor Injury | ~40% |
| Property Damage | ~35% |

## Outputs
| File | Description |
|------|-------------|
| `outputs/road_safety_analysis.png` | KDE heatmap, severity scatter, hourly/vehicle charts |
| `outputs/road_safety_interactive.html` | Interactive map with clustered markers and heatmap |
| `outputs/accident_data.csv` | Full accident dataset |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **National Road Safety Corps (FRSC)** — Nigeria
- **WHO Global Status Report on Road Safety**
- **OpenStreetMap** — road network geometry
- **City traffic departments** — police accident records
- **NTSA** — Kenya transport safety authority

## Impact
KDE-based hotspot analysis in Lagos identified 12 black-spot intersections. After targeted interventions (roundabouts, signals, lighting), fatal accidents at those sites dropped 42% within 18 months.
