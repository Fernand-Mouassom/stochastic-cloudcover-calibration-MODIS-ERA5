from netCDF4 import Dataset
from datetime import datetime, timedelta
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray
import glob
import os
import re

def find_matching_files(mod35_folder, mod06_folder, mod35_output_folder, mod06_output_folder):
    os.makedirs(mod35_output_folder, exist_ok=True)
    os.makedirs(mod06_output_folder, exist_ok=True)

    mod35_files = [f for f in os.listdir(mod35_folder) if f.endswith(".hdf")]
    mod06_files = [f for f in os.listdir(mod06_folder) if f.endswith(".hdf")]

    for mod35_file in mod35_files:
        mod35_key = ".".join(mod35_file.split(".")[1:4])

        mod06_file = next((f for f in mod06_files if ".".join(f.split(".")[1:4]) == mod35_key), None)

        if mod06_file:
            mod06_key = ".".join(mod06_file.split(".")[1:4])

            mod35_path = os.path.join(mod35_folder, mod35_file)
            mod06_path = os.path.join(mod06_folder, mod06_file)

            mod35_output_path = os.path.join(mod35_output_folder, f"MOD35_L2.{mod35_key}.hdf")
            mod06_output_path = os.path.join(mod06_output_folder, f"MOD06_L2.{mod06_key}.hdf")

            try:
                os.link(mod35_path, mod35_output_path)
                os.link(mod06_path, mod06_output_path)
                print(f"Matched and saved: {mod35_key} and {mod06_key}")
            except Exception as e:
                print(f"Error saving matched files for {mod35_key} and {mod06_key}: {e}")
        else:
            print(f"No matching MOD06_L2 file found for {mod35_file}. Skipping...")


def rename_file(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".hdf"):
            parts = filename.split(".")
            date_part = parts[1][1:]
            time_part = parts[2]
            year = int(date_part[:4])
            julian_day = int(date_part[4:])
            hour = int(time_part[:2])
            minute = int(time_part[2:])
            date = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
            new_filename = f"{prefix}.{date.strftime('%Y-%m-%d')}_{hour:02d}-{minute:02d}.061.cloud_mask_okay.hdf"
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")

def read_modis(file_path, var_name):
    try:
        hdf_file = SD(file_path, SDC.READ)
        data_obj = hdf_file.select(var_name)
        attrs = data_obj.attributes()
        data = data_obj[:]
        fillvalues = attrs.get("_FillValue", None)
        scale_factor = attrs.get("scale_factor", 1.0)
        offset = attrs.get("add_offset", 0.0)
        data = data * scale_factor + offset
        if fillvalues is not None:
            data = np.ma.masked_where(data_obj[:] == fillvalues, data)
        return data
    except Exception as e:
        return None

def low_cloud_extraction(file_folder35, file_folder06, region_bound, modis_output):
    r_latmin, r_latmax=region_bound[0]
    r_lonmin, r_lonmax=region_bound[1]
    filenames=sorted([f for f in os.listdir(file_folder35) if f.endswith(".hdf")])
    for fil35 in filenames:
        print(fil35)
        fil06=fil35.replace("MOD35_L2", "MOD06_L2")
        file_path=os.path.join(file_folder35, fil35)
        file_path06=os.path.join(file_folder06, fil06)
        try:
            hdf_file = SD(file_path, SDC.READ)
            latitude = hdf_file.select("Latitude").get()
            longitude = hdf_file.select("Longitude").get()
            match = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})\.(\d{3})",  file_path)
            year= int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            date = datetime(year, month, day, hour, minute)
            time = np.datetime64(date).astype('datetime64[ns]')
            lat_min, lat_max = np.min(latitude), np.max(latitude)
            lon_min, lon_max = np.min(longitude), np.max(longitude)
            hdf_file.end()
            if month in [1, 2, 3, 4, 5, 10, 11, 12]: # The monsoon months (June to September) were excluded change if needed
            ################ Zenith Angle Condition####################################

                data_sz = read_modis(file_path, "Sensor_Zenith")
                rows_sz, cols_sz = data_sz.shape[0], data_sz.shape[1]
                x_res_sz=np.linspace(lon_min, lon_max, cols_sz)
                y_res_sz=np.linspace(lat_min, lat_max, rows_sz)
                new_lon_sz, new_lat_sz = np.meshgrid(x_res_sz, y_res_sz)

                data_sz_array= xr.DataArray(data_sz, dims=["y", "x"], coords={"latitude": (["y", "x"], new_lat_sz), "longitude": (["y", "x"], new_lon_sz),})
                data_sz_array.rio.write_crs("EPSG:4326", inplace=True)

                angle_zenith_data=data_sz_array[:].values
                angle_zenith_data=angle_zenith_data.reshape(1, new_lon_sz.shape[0], new_lon_sz.shape[1])

                data_sz = xr.Dataset(data_vars={"sz": (["time", "latitude", "longitude"], angle_zenith_data)}, coords={"time": [time], "latitude": new_lat_sz[:, 0], "longitude": new_lon_sz[0, :],},)
                cropped_data_sz = data_sz.sel(latitude=slice(r_latmin, r_latmax), longitude=slice(r_lonmin, r_lonmax))
                cropped_data_sz_mask=np.where(cropped_data_sz.sz<=25, 1, 0)

        ################ Low Cloud Condition ###############################################

                if cropped_data_sz_mask.mean()>=0.8:
                    data_tp=read_modis(file_path06, "cloud_top_height_1km")
                    rows_tp, cols_tp = data_tp.shape[0], data_tp.shape[1]
                    x_res_tp=np.linspace(lon_min, lon_max, cols_tp)
                    y_res_tp=np.linspace(lat_min, lat_max, rows_tp)
                    new_lon_tp, new_lat_tp = np.meshgrid(x_res_tp, y_res_tp)

                    data_tp_array= xr.DataArray(data_tp, dims=["y", "x"], coords={"latitude": (["y", "x"], new_lat_tp), "longitude": (["y", "x"], new_lon_tp),})
                    data_tp_array.rio.write_crs("EPSG:4326", inplace=True)

                    low_cloud_data=data_tp_array[:].values
                    low_cloud_data=low_cloud_data.reshape(1, new_lon_tp.shape[0], new_lon_tp.shape[1])

                    data_tp=xr.Dataset(data_vars={"ctp": (["time", "latitude", "longitude"], low_cloud_data)}, coords={"time": [time], "latitude": new_lat_tp[:, 0], "longitude": new_lon_tp[0, :],},)
                    cropped_data_tp = data_tp.sel(latitude=slice(r_latmin, r_latmax), longitude=slice(r_lonmin, r_lonmax))

                    cropped_data_tp_mask=np.where(cropped_data_tp.ctp>2500, 1, 0)

            ################ Cloud Mask extraction ###############################################

                    if cropped_data_tp_mask.mean()<=0.2:
                        data_cm = read_modis(file_path, "Cloud_Mask")
                        first_byte = data_cm[0, :, :]
                        first_byte = first_byte.astype('uint8')
                        rows_cm, cols_cm = data_cm.shape[1], data_cm.shape[2]
                        x_res_cm=np.linspace(lon_min, lon_max, cols_cm)
                        y_res_cm=np.linspace(lat_min, lat_max, rows_cm)
                        new_lon_cm, new_lat_cm = np.meshgrid(x_res_cm, y_res_cm)

                        data_cm_array = xr.DataArray(first_byte, dims=["y", "x"], coords={"latitude": (["y", "x"], new_lat_cm), "longitude": (["y", "x"], new_lon_cm),})
                        data_cm_array.rio.write_crs("EPSG:4326", inplace=True)

                        cloud_mask_data=data_cm_array[:].values
                        cloud_mask_data=cloud_mask_data.reshape(1, new_lon_cm.shape[0], new_lon_cm.shape[1])

                        data_cm = xr.Dataset(data_vars={"cm": (["time", "latitude", "longitude"], cloud_mask_data)}, coords={"time": [time], "latitude": new_lat_cm[:, 0], "longitude": new_lon_cm[0, :],},)
                        cropped_data_cm = data_cm.sel(latitude=slice(r_latmin, r_latmax), longitude=slice(r_lonmin, r_lonmax))

                        # cloud_mask_flag = (cropped_data_cm & 0b00000001) >> 0
                        # land_water_flag = (cropped_data & 0b11000000) >> 6
                        # shadow_flag = (cropped_data & 0b10000000000) >> 10

                        sun_glint_flag = (cropped_data_cm & 0b10000) >> 4

                        if sun_glint_flag.cm.mean().values>=0.9:

                            confidence_level = (cropped_data_cm & 0b00000110) >> 1
                            cloud_fraction=confidence_level.cm.values.astype(float)

                            replacement_map = {0: 1, 1: 0.67, 2: 0.27, 3: 0}

                            for i in range(cloud_fraction.shape[1]):
                                cloud_fraction[0, i, :]=[replacement_map.get(x, x) for x in cloud_fraction[0, i, :]]


                            lat=cropped_data_cm.latitude.values
                            lon=cropped_data_cm.longitude.values

                            cloud_fraction_data=xr.Dataset(data_vars={"CAF": (["time", "latitude", "longitude"], cloud_fraction)}, coords={"time": [time], "latitude":lat, "longitude":lon})

                            file_caf=fil35.replace("MOD35_L2", "CAF")
                            file_caf=file_caf.replace("cloud_mask_okay.hdf", "cloud_fraction.nc")
                            os.makedirs(modis_output, exist_ok=True)
                            file_caf_path=os.path.join(modis_output, file_caf)

                            cloud_fraction_data.to_netcdf(file_caf_path)

                            print(f"Cloud fraction for file {fil35} successfully extracted and saved as {file_caf}")

                        else:
                            print(f"Sun Glint detected for file {fil35} Skipping ...")
                    else:
                        print(f"High cloud detected for file {fil35} Skipping ...")
                else:
                    print(f"Sensor far from study area for file {fil35} Skipping ...")

        except Exception as e:
            continue

def extract_modis_time_step_in_era5(input_dir, era5_data_main_file, prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files=os.listdir(input_dir)
    era5_data_main=xr.open_dataset(era5_data_main_file)
    for modis_file in files:
        try:
            modis_path = os.path.join(input_dir, modis_file)
            era5_file = modis_file.replace("CAF", "ERA5")
            era5_file = era5_file.replace("cloud_fraction", prefix)
            era5_path = os.path.join(output_dir, era5_file)
            modis_data=xr.open_dataset(modis_path)
            times=modis_data.time.values
            era5_data = era5_data_main.sel(time=times)
            era5_data.to_netcdf(era5_path)
            print(f"The timestep {times} have been processed and saved in {era5_path}")
        except Exception as e:
            print(f"Error processing {modis_file} : {e}")
            continue


def csv_file_contruction(input_dir_modis, input_dir_era5t, input_dir_era5q, input_dir_era5p, csv_file):
    filenames=[f for f in os.listdir(input_dir_modis) if f.endswith(".nc")]
    results_df=pd.DataFrame({"Times":[0], "T":[0], "q":[0], "sp":[0], "CAF":[0]})

    for caf_file in filenames:
        try:
            era_file=caf_file.replace("CAF", "ERA5")
            temp_file=era_file.replace("cloud_fraction", "temperature")
            sp_file=era_file.replace("cloud_fraction", "specific_humidity")
            p_file=era_file.replace("cloud_fraction", "surface_pressure")

            caf_path=os.path.join(input_dir_modis, caf_file)
            temp_path=os.path.join(input_dir_era5t, temp_file)
            sp_path=os.path.join(input_dir_era5q, sp_file)
            p_path=os.path.join(input_dir_era5p, p_file)

            caf_data=xr.open_dataset(caf_path)
            temp_data=xr.open_dataset(temp_path)
            sp_data=xr.open_dataset(sp_path)
            p_data=xr.open_dataset(p_path)

            p_var=list(p_data.data_vars)[0]

            results_df1=pd.DataFrame({"Times":caf_data.time.values, "T":[temp_data.t.mean().values], "q":[sp_data.q.mean().values], "sp":[p_data[p_var].mean().values] "CAF":[caf_data.CAF.mean().values]})

            results_df = pd.concat([results_df, results_df1], ignore_index=True)

        except Exception as e:
            continue
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")
