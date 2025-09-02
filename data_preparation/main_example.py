from functions import find_matching_files
from functions import rename_file
from functions import read_modis
from functions import low_cloud_extraction
from functions import extract_modis_time_step_in_era5
from functions import csv_file_contruction
from functions import add_surface_pressure_to_csv_file
from functions import add_min_max_to_csv_file


file_folder35="../data/modis_data/MOD35_L2_data" #Path from MOD35_L2 data files
file_folder06="../data/modis_data/MOD06_L2_data" #Path from MOD06_L2 data files
matched_folder35="../data/modis_data/MOD35_06_matched_data" #Path where to store MOD35_06_matched data files
matched_folder06="../data/modis_data/MOD06_35_matched_data" #Path where to store MOD06_35_matched data file
modis_caf_output_dir=f"../data_okay{n}/modis_caf" #Path where to store CAF data file for each region_bound or grid cell
era5_temp_output_dir=f"../data_okay{n}/era5_temp" #Path where to store temperature data file for each region_bound or grid cell
era5_sp_output_dir=f"../data_okay{n}/era5_sp"    #Path where to store specific humidity data file for each region_bound or grid cell
era5_press_output_dir=f"../data_okay{n}/era5_press"   #Path where to store surface pressure data file for each region_bound or grid cell


era5_temp_main_file=f"../data/era5_data/temperature_vmean_interpolated_rounded/era5_temperature_vmean_interpolated_rounded_merged{n}.nc" #Path from ERA5 temperature data
era5_sp_main_file=f"../data/era5_data/specific_humidity_vmean_interpolated_rounded/era5_specific_humidity_vmean_interpolated_rounded_merged{n}.nc" #Path from ERA5 specific_humidity
era5_surface_pressure_main_file=f"../data/era5_data/surface_pressure/surface_pressure_interpolated_rounded{n}.nc" #Path from ERA5 surface pressure

csv_file = f"../data_okay/data_okay{n}/modis_and_netcdf_data{n}.csv" #Path where to store output csv file for each region_bound or grid cell


matchin_time=find_matching_files(file_folder35, file_folder06, matched_folder35, matched_folder06)
rename_file(matched_folder35, "MOD35_L2")
rename_file(matched_folder06, "MOD06_L2")

region_bounds = [
    [[-6.0, -5.75], [-1.0, -0.75]],
    [[-7.0, -6.75], [-1.0, -0.75]],
    [[-5.0, -4.75], [ 1.0,  1.25]],
    [[-4.0, -3.75], [ 2.0,  2.25]],
    [[-5.5, -5.25], [ 0.5,  0.75]],
    [[-6.0, -5.75], [ 0.0,  0.25]],
    [[-6.5, -6.25], [-0.5, -0.25]],
]

N=len(region_bounds)

for n in range(1, N):
  low_cloud_extraction(matched_folder35, matched_folder06, region_bounds[n], modis_caf_output_dir)
  extract_modis_time_step_in_era5(modis_caf_output_dir, era5_temp_main_file, "temperature", era5_temp_output_dir)
  extract_modis_time_step_in_era5(modis_caf_output_dir, era5_sp_main_file, "specific_humidity", era5_sp_output_dir)
  extract_modis_time_step_in_era5(modis_caf_output_dir, era5_surface_pressure_main_file, "surface_pressure", era5_sp_output_dir)
  csv_file_contruction(modis_caf_output_dir, era5_temp_output_dir, era5_sp_output_dir, era5_press_output_dir, csv_file)


