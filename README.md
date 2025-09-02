## VIMFC as predictor of Extreme Precipitation Events

This repository contains the code and data used in the paper:
**"Convolutional Neural Network-Based Insights into Extreme Precipitation Regional Dynamics over Central Africa Using Moisture Flux Patterns"**
*Fernand L. Mouassom · Alain T. Tamoffo · Elsa Cardoso-Bihlo, JGR: Atmosphere, 2025*

If you find meaningful errors in the code or have questions, please contact Fernand Mouassom 

## Organization of repository 
* **input_data**: input data used for analysis (not all raw data is included due to size; see details below)
* **notebooks**: jupyter notebooks and python scripts to read and analyze data, and create figures
* **ncl_code** : directory for codes used to calculate vertically integrated moisture flux convergence (VIMFC), and the stream function for the Congo basin cell intensity. This was done using **ncl** for reasons of efficiency and speed compared to **Python**.
* **processed_data**: processed data from analysis
* **figures** :  directory for figure png created by running figure notebooks in **notebooks** directory
* environment.yml : specifies python packages needed to run notebooks
* environment_LRP.yml : specifies python packages needed to run the LRP notebook

## data
All the data analyzed in the paper is publicly available at the ERA5 reanalysis dataset, produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) as part of the Copernicus Climate Change Service (C3S) : [ERA5 dataset on Copernicus for Precipitation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) and [ERA5 dataset on Copernicus for other variables](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)

This include:

* **Daily precipitation** : (pr);
* **Daily Specific humidity** : (q);
* **Daily Surface pressure** : (sp);
* **Daily Zonal wind** : (u); 
* **Daily Meridional wind** (v).
 
Due to the large size of the raw data, they have not been uploaded in this repositry 

## Steps to run the notebooks:
1. download this repository  
2. download all the variables listed  above from ERA5 from 1984-2023 (the variables q, u, v have to be dowloaded for the pressure level available on ERA5. 
3. use the ncl codes provided into **ncl_code** directory to compute the vertically Integrated moisture flux convergence and the stream fuction for the congo basin cell intensity
4. install the required python modules using conda or pip. The environment.yml and environment_LRP.yml provide information on the required modules. (The environment_LRP.yml files specifies the tensorflow 1 compatible environment needed to calculate the layerwise relevance propagation - see **notebooks** directory for more details) 
```bash
pip install -e . --user
```
5. run and/or edit the notebooks. ////////////////
