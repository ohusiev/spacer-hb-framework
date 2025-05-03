# ⚡ SPACER-HB Framework  
**Spatial Pre-feasibility Assessment for Community Energy, Renovation, and Household Benefits**

---

## What is SPACER-HB?

SPACER-HB is an ad-hoc, spatial, and public-data-based framework for early-stage local energy community planning. It integrates rooftop solar and renovation potential with household-level energy vulnerability mapping to support informed pre-feasibility decisions.


## Main purpose of the framework:
1. To provide a simple, adaptable tool that helps in spatial assessment of the potential for local energy communities. It uses publicly available data and open-source tools to estimate building energy demand, renovation impacts, and solar rooftop potential. 
2. The framework combines statistical census data (LAU2, LAU1) with building-level information in a unified data workflow to estimate community solar self-consumption, average household savings, and the effects of various pricing rules—using only publicly available data and no private metering.

```
The framework is designed for prefeasibility analysis, aiming to make a straightforward ad-hoc workflow for potential municipalities and/or researchers to explore energy scenarios and identify priority areas from spatial perspective, without needing detailed local data or complex models.
```

## Content
 1. Workflow, modules and python libraries
 2. Input parameters file and naming standartization
 3. Output
 4. Interpretation Output
 5. Demo


## 1. Workflow, modules and python libraries

### 1.1 Python libraries and Anaconda environment:
​    Python ver used: 3.9.16
​    Anaconda environment:
​    pip installed pachages:
### 1.2 QGIS and Whitebox plugin:
​    QGIS version:
​    Whitebox Tools:
​    More on how to instal WhiteboxTools plugin in QGIS:
### 1.3 LoadProfileGenerator Tool

## 2. Input parameters file and naming standartization

### 2.1 Set-up directory tree and environment 

> **Workig directory set-up by default built in a simple way, i.e it is assumed to be a single project (i.e: single Case Study per root directory tree).**

> Single action for a new project, if needed.

### 📁 Project Directory Structure

```text
root_folder/
│
├── lidar/
│   └── Contains raw LIDAR files for the area of interest. These files are the input for further spatial processing via Whitebox Tools: Rooftop Analysis.
│
├── pyqgis_whitebox_tool/
│   └── Stores output from LIDAR processing scripts run via the PyQGIS console.
│
├── raster/
│   └── Holds supporting raster datasets (e.g., land cover, elevation models) used in analysis.
│
├── vector/
│   └── Contains project-related vector files, including:
│       - Building footprints
│       - Initial outputs from Whitebox Tools: Rooftop Analysis
│
├── LoadProGen/
|    └── Includes electricity load profiles for households. These are used for energy demand
|          analysis within the framework.
|
└── data/
    └── Stores temporary and key steps data files
```


### 2.2 Placing files into default directory path

### 2.3 FILES and FIELDS naming standartization



### Modules
​    Module 1: ---
​    Module 2: 
​        Pyqgis and WhiteBox Rooftop analysis automatiser
​        pivot of rooftop data
​        geopandas facade analyser
​    Module 3: 
​        Rooftop PV analysis
​    Module 4: 
​        dwelling profile generator and assigner
​        self-consumption estimator
​    Module 5: 
​        economic calculation
​    Module 5: 
​        energy community estimator
​    Module 6: 
​        Visualization

## Input parameters file and naming standartization
