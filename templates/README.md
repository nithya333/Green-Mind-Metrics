# 🌿 Green Mind Metrics: Eco-Psychology Analytics Dashboard

**A Data Science & Web Engineering solution correlating geospatial environmental quality with student mental well-being across university campuses.**

Team Members: Krupa P Nadgir, Pavithra N, Manasa S

## Overview

Green Mind Metrics is an exploratory research project that investigates the "Green Paradox": _Do greener campuses actually house happier students?_

By merging **15-band Sentinel-2 Satellite Imagery** (Geospatial) with **Student Psychometric Surveys** (Quantitative Psychology), this platform calculates unique indices like **Green Score** (Ecological Health) and **Mind Score** (Student Well-being) to visualize the impact of built environments on academic stress.


## Project Overview

Urban university campuses are experiencing rapid physical expansion, often at the cost of green cover. While the link between nature and mental health is well-documented, campus administrators lack a **unified, data-driven framework** to measure this relationship locally.

**Green Mind Metrics** solves this by merging two distinct data pipelines:

1. **Geospatial:** Sentinel-2 Satellite imagery (NDVI, LST, NDBI).
2. **Psychometric:** Primary survey data measuring student stress, mood, and focus.

The result is a unified dashboard that calculates a **"GreenScore"** (Ecological Health) and a **"MindScore"** (Student Well-being) to identify high-stress "concrete jungles" vs. restorative "green oases."

## Objectives

As outlined in the Project Report:

1. **Quantify Ecology:** Map campus greenery using 15-band satellite features (NDVI, GNDVI, LST).
2. **Assess Well-being:** Collect primary data on student stress, mood, and perceived restorativeness.
3. **Correlate:** Identify statistical relationships between "Green Cover" and "Acute Stress."
4. **Visualize:** Build an interactive comparison dashboard for stakeholders to optimize campus planning.

## Key Features

- ** Interactive Dashboard:** A split-pane web application allowing side-by-side comparison of up to 5 campuses.
- ** 15-Band Satellite Analysis:** Real-time rendering of NDVI, NDBI, LST (Land Surface Temp), and LULC (Land Use) masks.
- ** Psychometric Integration:** Correlates environmental data with survey metrics like _Perceived Restorativeness_, _Acute Stress_, and _Mood_.
- ** Dynamic Graphing:** On-the-fly chart generation using Chart.js to compare specific metrics (e.g., _Concrete % vs. Stress Level_).
- ** Heatmap Generation:** Python backend generates thermal and vegetation heatmaps from raw TIF files on demand.


## Data Sources

* **Geospatial Data:** Fetched from **Sentinel-2 (Level 2A)** via Sentinel Hub.
* *Resolution:* 10m.
* *Metrics:* NDVI (Vegetation), NDBI (Built-up Index), LST (Land Surface Temp), LULC (Land Use Masks).


* **Psychometric Data:** Primary data collected via Google Forms (n=100+).
* *Metrics:* Perceived Stress Scale (PSS), Mood (1-10), Focus improvement in green areas.


## Tech Stack

| Component         | Technology                | Description                                                           |
| ----------------- | ------------------------- | --------------------------------------------------------------------- |
|   Backend         |   Python (Flask)          | API handling, data merging (Pandas), and image processing (Rasterio). |
|   Frontend        |   HTML5 + Alpine.js       | Reactive UI logic without the bloat of React/Vue.                     |
|   Storytelling    |   Tailwind CSS            | Modern, responsive, and dark-mode-first design.                       |
|   Geospatial      |   Rasterio & Matplotlib   | Processing TIF satellite bands into viewable heatmaps.                |
|   Visualization   |   Chart.js                | Rendering interactive bar and comparison charts.                      |


## Data Pipeline

1. **Ingestion:**

- **Geospatial:** Sentinel-2 Level-2A imagery fetched via Sentinel Hub.
- **Psychometric:** Google Forms responses (n=100+) cleaning and normalization.

2. **Processing:**

- Calculation of **NDVI** (Vegetation), **NDBI** (Built-up), and **LST** (Temperature).
- Conversion of Likert scale survey responses to numerical "Stress/Mood Scores".

3. **Visualization:**

- Backend serves processed PNGs from raw TIFs.
- Frontend fetches JSON stats and renders comparison graphs.

## Analysis & Insights

- **Correlation Matrix:** Analyzing relationships between LST and Stress.
- **Hypothesis Testing:** "Does higher Green Cover (NDVI) correlate with lower acute stress?"
- **Outlier Detection:** Identifying campuses that break the trend.
* **The Green Paradox:** Some campuses with high green cover still report high stress due to *lack of access* (greenery is fenced off).
* **Temperature Correlation:** Campuses with >40% concrete cover show a +2°C higher Land Surface Temperature (LST), correlating with lower mood scores.

## The Dashboard

* **Split-Pane Comparison:** Compare up to 5 colleges side-by-side with synchronized scrolling.
* **Dynamic Graphing:** Select specific metrics (e.g., *NDVI vs. Stress*) to generate instant bar charts.
* **Layer Toggling:** Switch between **True Color Satellite**, **LULC Masks**, **NDVI Heatmaps**, and **Thermal (LST)** layers.
* **Calculated Indices:**
* `Green Score`: Weighted average of Vegetation density and Canopy health.
* `Mind Score`: Inverse function of Stress levels and Mood reports.
