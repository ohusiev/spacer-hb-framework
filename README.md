# ⚡ SPACER-HB Framework  
**Spatial Pre-feasibility Assessment for Community Energy, Renovation, and Household Benefits**

---

## What is SPACER-HB?

SPACER-HB is an ad-hoc, spatial, and public-data-based methodological framework for early-stage local energy community planning. It integrates solar rooftop and buildings' facade renovation potential inventory with household-level energy vulnerability mapping to support informed pre-feasibility decisions.


## Main purpose of the framework:
1. To provide an adaptable set of custom tools that help in spatial assessment of the potential for local energy communities. It utilizes publicly available data and open-source tools to estimate building energy demand, the impacts of renovations, and solar rooftop potential. 
2. The framework combines statistical census data (LAU2, LAU1) with building-level information in a unified data workflow to estimate community solar self-consumption, average household savings, and the effects of various pricing rules—using only publicly available data and no private metering.

```
The framework is designed for prefeasibility analysis, aiming to make a straightforward ad-hoc workflow for potential municipalities and/or researchers to explore energy scenarios and identify priority areas from spatial perspective, without needing detailed local data or complex models.

The methodologogical workflow presents a custom combinaiton of tools and methods aligned together to establish a workflow for part of work wihtin a PhD disseration titled "Exploring and Supporting Energy Communities through Spatially Integrated Methods in Local Energy Planning".

The current framework, in short, depicts components integrating census-level statistical boundaries with building-specific, aligning tabular data with geospatial vector and raster data. For that a python based automatization scripts has been developed to streamline certain aspects. While the intentin didn't cover a front end/ backend technical development, as well as integration of Database. The set of refined software integrated solutions has benn existing in open source foem, requiring either detailed amount of data, or from the point of comertial solutions. 

In contrast, while acknowleging the variability the tools, an dbeing the matter of the researhc in the disruptiave public availability of generative Ai services that bring an options to combine/introduce abovedescribed generic tools/methods into variaty of potetial ad-hoc workflows, adjustable for any kind of purpose. The detailed description of the workflow with pythons scripts for partial automatisation and facilitation of the process is intended to serve as a reproducable and yet realistic tutorial that might be of use for the potential reasearcher/ municipality(energy analyst) and other stakeholders, or energy community intusiast, seaking to build data pipline, with exploring particual approach for mapping of energy vulnerability and socio-economic potential of building stock energy renovation.

The certain estimation and simplification has been described and systematised for parameters and variables, done through manimulation of data in tabular form (MS Excel), while intermediate input/output files considered as of xlsx, csv and geojson formats, primarily processed by the python libraries as pandas and geopandas, as well as QGIS software for furtehr visualization.

DESCLAMER: This Framework initially is not aimed to be optimised tool from the point of performance, used software modelues and developed code scripts, and was only elementary tested for whe components of software lifecycle (steps covered with manual check on mistakes and bugs starting from requirements definition, system specification, system and its units design, unit and system coding, and manual systems integration testing and only primary iterative checks for a principal components to adresse system validation testing.

Also, the necessary presision of the calculations was intended to be primary focus, but rather as a semi-authomatised workflow, as it was already mentioned for a pre-feasibility estimation with extended level of spatial scope and at the same time.
The stop point for this framework initially considered establishd pipeline with a final semi-automatised dasboard creation, however this final aspect is under the development.
While the public announecement of AI generative tool has marked a new era with the incorporation of LLMs into GIS, this framework also demonstrates approach for utilisation of it with sertain degree for aoutomation of preliminary data cleaning and developing a scripts that before rather could take an immense constraint for beginner and take a considerable amount of time for a most of mid-to high level of GIS analysis and researchers. The processing structure implementation and application of spatial data will be revolutionized by automation, improved accessibility.

```

## Content
 1. Workflow, setup environment for python libraries 
 2. Structure, Input parameters file and naming standartization
 3. Python scripting developed as 'Modules' 
 4. Interpretation Output
 5. Demo


## 1. Workflow and setup for python libraries

### 1.1 Python libraries and Anaconda environment:
​    Python ver used: 3.9.16
​    Anaconda environment:
​    pip installed pachages:
### 1.2 QGIS and Whitebox plugin:
​    QGIS version:
​    Whitebox Tools:
​    More on how to instal WhiteboxTools plugin in QGIS:
### 1.3 LoadProfileGenerator Tool

## 2. Structure, Input parameters and naming standartization


IMPORTANT: It is important to ensure the same coordinate system from the beginning!
### 2.1 Set-up directory tree 

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
|    └── Includes electricity load profiles for households. These are used for energy demand analysis within the framework.
|
└── data/
    └── Stores temporary and Python scripting Modules' output data files
```


### 2.2 Placing files into default directory path

### 2.3 FILES and FIELDS naming standartization

Attributes for segments of building rooftop analysis layer
| #  | Attribute Name                                             | Short Name             |
|----|-----------------------------------------------------------|------------------------|
|    | **IDs for levels analysis interplay**                     |                        |
| 1  | Segment ID                                                | segm_id                |
| 2  | Building ID                                               | build_id               |
| 3  | Census ID                                                 | census_id              |
|    | **Primary, WhiteBox RooftopAnalysis**                     |                        |
| 4  | Slope, degree                                             | slope                  |
| 5  | Aspect (angle), degree                                    | aspect                 |
| 6  | Segment area (projected on horizontal surface), m²        | s_area                 |
| 7  | Maximum elevation of segment, m                           | max_elev               |
|    | **Secondary, WhiteBox RooftopAnalysis**                   |                        |
| 9  | WhiteBox rooftop analysis building ID                     | wb_build_id            |
| 10 | Hillshade of Segment                                      | hilshade               |
|    | **Supportive**                                            |                        |
| 8  | Exception, (1 -excluded)                                  | exception              |
| 11 | Building use type [residential label]                     | building               |
| 12 | Segment useful area, m²                                   | s_usf_area             |
| 14 | Segment center X, EPSG:                                   | crs[%epsg:num]_x       |
| 15 | Segment center Y, EPSG:                                   | crs[%epsg:num]_y       |
|    | **Optional**                                              |                        |
| 16 | Building address, street                                  | build_str              |
| 17 | Building address, number                                  | build_num              |
| 18 | Building center X, EPSG:                                  | b_cent_x               |
| 19 | Building center Y, EPSG:                                  | b_cent_y               |


Table Attributes of solar rooftop output analysis
| #  | Attribute Name                                 | Short Name   |
|----|------------------------------------------------|--------------|
| 1  | Building ID                                    | build_id     |
| 2  | Census ID                                      | census_id    |
| 3  | Number of solar PV panels                      | n_panels     |
|    | *Number of solar thermal panels                |              |
| 4  | Installed capacity of solar PV panels, kWp     | panel_kWp    |
|    | *Installed capacity of solar thermal panels, kWp|             |
| 5  | *Monthly generation (kWh) m = 1, [1,12], m∈R   | [1:12]       |
| 6  | Annual generation, kWh                         | Total, kWh   |
|    | *for PV or ST 100%                             |              |

Note: * reflects the group of attributes that will hold different values considering if initially solar PV or solar thermal technology estimation was accepted


## 3. Python script as Modules 
​ **Module 1: PyQGIS and WhiteBox Rooftop analysis automatiser**
        
        00_pyqgis_wb_automator.py
        mod_01_pivot_rooftop_data.py
        ---
​    
**Module 2: Rooftop PV analysis**
        
        mod_02_pv_power_month.py

**Module 3:  Calculate Simple geometry of buildingas based on 2d footprint and height**

        mod_03_geopandas_facade_analyser.py

What it is about: Joining non-spatial data (CSV) to a GeoDataFrame; Calculating new columns

**Module 4: Dwelling profile generator and assigner, self-consumption estimator**
        
        mod_04_ener_consum_profile_assigner.py
        mod_04_energy_profile_aggregation.py
        mod_04_self_consump_estimation.py
​   
**Module 5: Building stock heating/cooling demand/consumption assignment and Economic calculation**

        mod_05_inspire_db_assigner.py from https://zenodo.org/records/3256270
        mod_05_1_simple_kpi_calc.py
        mod_05_2_test_economic_analysis.py

**Module 6: Energy community estimator**

        mod_06_enercom_estimator.py

**Module 7: Visualization**

        !!! mod_07_self-cons_scenarios_calc.py
        mod_07_geo_visualization.py
        QGIS +filters

