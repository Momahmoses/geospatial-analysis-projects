# Geospatial Data Analysis Projects

10 professional geospatial analysis projects addressing real-world challenges — from disaster response to food security. Each project uses publicly available datasets, Python geospatial libraries, and produces publication-quality maps and insights.

## Projects

| # | Project | Key Libraries | Impact Area |
|---|---------|--------------|-------------|
| 01 | [Urban Heat Island Mapping](./01-urban-heat-island/) | GeoPandas, Matplotlib, Folium | Climate & Public Health |
| 02 | [Flood Risk & Early Warning](./02-flood-risk/) | GeoPandas, Shapely, Folium | Disaster Preparedness |
| 03 | [Healthcare Accessibility](./03-healthcare-access/) | GeoPandas, NetworkX, Folium | Public Health |
| 04 | [Food Desert Identification](./04-food-desert/) | GeoPandas, Scikit-learn, Plotly | Food Security |
| 05 | [Deforestation Monitoring](./05-deforestation/) | Rasterio, NumPy, Matplotlib | Environment |
| 06 | [Road Safety Hotspot Analysis](./06-road-safety/) | GeoPandas, Scikit-learn, Folium | Transportation Safety |
| 07 | [Air Quality & Pollution Mapping](./07-air-quality/) | GeoPandas, Plotly, Folium | Environmental Health |
| 08 | [Crop Yield Optimization](./08-crop-yield/) | NumPy, Rasterio, Matplotlib | Agriculture |
| 09 | [Disaster Response Routing](./09-disaster-response/) | GeoPandas, NetworkX, Folium | Emergency Management |
| 10 | [Public Transit Gap Analysis](./10-transit-gap/) | GeoPandas, Folium, Plotly | Urban Mobility |

## Tech Stack

- **Python 3.10+**
- **GeoPandas** — vector geospatial data
- **Rasterio** — raster/satellite data
- **Folium** — interactive web maps
- **Plotly** — interactive charts
- **Matplotlib / Seaborn** — static publication charts
- **Scikit-learn** — spatial clustering & ML
- **Shapely** — geometric operations
- **NetworkX** — routing & network analysis

## Setup

```bash
pip install geopandas folium plotly rasterio shapely networkx scikit-learn matplotlib seaborn pandas numpy requests
```

Each project folder contains:
- `analysis.py` — main analysis script
- `notebooks/` — Jupyter notebook walkthrough
- `outputs/` — generated maps and charts
- `data/` — data files or download scripts
- `README.md` — project documentation

## Data Sources

Projects use publicly available data from:
- [OpenStreetMap](https://www.openstreetmap.org/) via OSMnx
- [NASA Earth Observations](https://neo.gsfc.nasa.gov/)
- [World Bank Open Data](https://data.worldbank.org/)
- [GADM Administrative Boundaries](https://gadm.org/)
- [Global Health Observatory (WHO)](https://www.who.int/data/gho)
- [Global Forest Watch](https://www.globalforestwatch.org/)
- [OpenAQ](https://openaq.org/) — air quality
- [ReliefWeb](https://reliefweb.int/) — disaster data

---

*Built for real-world impact — each analysis targets a measurable improvement in human wellbeing.*
