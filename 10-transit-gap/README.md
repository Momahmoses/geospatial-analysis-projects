# Project 10 — Public Transit Gap Analysis

## Overview
Analyses transit route coverage against population density, job accessibility, income levels, and car ownership to identify corridors where people most need — but lack — public transport. Identifies top candidate corridors for new route investment.

## Methodology
1. **Demand model** — Population × (1 - car_ownership) × log(job_access)
2. **Transit access score** — Proximity to existing stops, decay by walking distance
3. **Gap score** — `demand × (1 - transit_access/100)`
4. **Gap classification** — Critical / High / Moderate / Low / Well Served
5. **Route candidates** — Top-10 zones by gap score with equity weighting
6. **Equity analysis** — Income vs access scatter with equity threshold lines

## Transit Gap Criteria
A zone qualifies as a **Critical Gap** when:
- Gap score ≥ 60th percentile of demand-weighted cells
- Car ownership < 30% (population depends on transit)
- Transit access score < 30/100

## Outputs
| File | Description |
|------|-------------|
| `outputs/transit_gap_map.png` | 4-panel: population+routes, access, demand, gap classification |
| `outputs/transit_gap_analysis.png` | Population by gap level + income vs access scatter |
| `outputs/transit_gap_interactive.html` | Interactive map with gap zones and route candidates |
| `outputs/transit_gap_data.csv` | Full per-cell dataset |

## Run
```bash
python analysis.py
```

## Real-World Data Sources
- **OpenStreetMap** — bus routes, stops, road network
- **GTFS feeds** — General Transit Feed Specification (schedule data)
- **WorldPop** — gridded population
- **OpenDataSoft** — city mobility datasets
- **World Bank** — vehicle registration / car ownership rates

## Impact
Transit gap analysis informed Nairobi's 2019 Bus Rapid Transit (BRT) corridor selection, prioritising routes through Mathare and Eastlands — areas where 85% of residents rely on matatus with no alternative.
