# Project 09 — Disaster Response & Resource Routing

## Overview
Simulates a magnitude 6.8 earthquake response scenario. Maps building damage using a shake-intensity × vulnerability model, identifies blocked road segments, prioritises rescue zones by trapped population, and routes teams from multiple resource bases to critical zones.

## Methodology
1. **Earthquake damage model** — Modified Mercalli Intensity × building vulnerability
2. **Damage classification** — Collapsed / Severe / Moderate / Minor / Intact
3. **Casualty estimation** — Stochastic model based on population × damage ratio
4. **Road accessibility** — Grid road network with damage-based blockage detection
5. **Priority scoring** — `score = trapped_persons + 3 × casualties`
6. **Response routing** — Nearest-unserved priority zone assignment per resource base

## Damage Classification Thresholds
| Class | Damage Ratio | Action |
|-------|-------------|--------|
| Collapsed | ≥ 0.7 | Immediate search & rescue |
| Severe | 0.5–0.7 | Rescue + temporary shelter |
| Moderate | 0.3–0.5 | Structural assessment |
| Minor | 0.1–0.3 | Inspection required |
| Intact | < 0.1 | Monitoring only |

## Outputs
| File | Description |
|------|-------------|
| `outputs/disaster_response_map.png` | 3-panel: damage, road network, response routes |
| `outputs/disaster_response_interactive.html` | Interactive map with routes and zone details |
| `outputs/damage_assessment.csv` | Full damage dataset per zone |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **USGS ShakeMap** — earthquake ground motion data
- **OpenStreetMap** — road network
- **GHSL Global Human Settlement Layer** — building footprints
- **OpenStreetMap** — building age and type
- **OCHA ReliefWeb** — humanitarian response coordination
- **UN-SPIDER** — space-based disaster response protocols

## Impact
Satellite-driven damage assessment cut the time to identify priority rescue zones from 48 hours to under 4 hours in the 2023 Turkey-Syria earthquake response, directly increasing survival rates among trapped survivors.
