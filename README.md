## Calibration of a Stochastic Model for Cloud Cover Using MODIS and ERA5 Data

This repository contains the code and data used in the paper:
**"Clouds of steel: The ferromagnetic behaviour of low clouds over the Gulf of Guinea"**
*Nyuydini Mohammed Kiven, Fernand L. Mouassom, Elsa Cardoso-Bihlo, Alex Bihlo and Boualem Khouider, npj Climate and Atmospheric Science, 2025*

If you find meaningful errors in the code or have questions, please contact Fernand L. Mouassom 

## Organization of repository 
* **input_data**: to download MODIS data run the code **download_modis.py** located in the folder **data_preparation** 
* **notebooks**: jupyter notebooks and python scripts to read and analyze data, and create figures
* **ncl_code** : directory for codes used to calculate vertically integrated moisture flux convergence (VIMFC), and the stream function for the Congo basin cell intensity. This was done using **ncl** for reasons of efficiency and speed compared to **Python**.
* **processed_data**: processed data from analysis
* **figures** :  directory for figure png created by running figure notebooks in **notebooks** directory
* environment.yml : specifies python packages needed to run notebooks
* environment_LRP.yml : specifies python packages needed to run the LRP notebook

## MODIS data
The two MODIS products used in this project, **MOD35_L2** and **MOD06_L2** can be downloaded from [Earthaccess](https://ladsweb.modaps.eosdis.nasa.gov)
## ERA5 data
All the data analyzed in the paper is publicly available at the ERA5 reanalysis dataset, produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) as part of the Copernicus Climate Change Service (C3S) : [ERA5 dataset on Copernicus for Precipitation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) and [ERA5 dataset on Copernicus for other variables](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)

This include for the Planetary Boundary Layer (1000 - 850 hPa):

* **Hourly temperature** : (t);
* **Hourly Specific humidity** : (q);
* **Hourly Surface pressure** : (sp);
 
Due to the large size of the raw data, they have not been uploaded in this repositry 

## Steps to run the notebooks:
1. download this repository  
2. download all the variables listed  above (ERA5 and MODIS). 
3. use the .py codes in the different folders
4. install the required python modules using conda or pip. The environment.yml rovide information on the required modules. 
```bash
pip install -e . --user
```
5. run and/or edit the .py files ///////
