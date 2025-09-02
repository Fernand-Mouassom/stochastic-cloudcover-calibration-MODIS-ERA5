import os
import earthaccess

def download_modis_data(product_id, start_date, end_date, bounding_box, output_folder):
    results = earthaccess.search_data(
        short_name=product_id,
        temporal=(start_date, end_date),
        bounding_box=bounding_box
    )

    downloaded_files = earthaccess.download(results, output_folder)
    return downloaded_files

auth = earthaccess.login()
product_id = "MOD06_L2"
start_date = "2000-01-01" # Change according to your study periode
end_date = "2024-12-31"  # Change according to your study periode
bounding_box = (-1, -7, 3, -3)  # Change according to your location
product_id = ["MOD35_L2", "MOD06_L2"]
output_folder = ["../MOD35_L2_data", "../MOD06_L2_data"] # Change as for your convenience 
for k in range(len(product_id)):
    downloaded_files = download_modis_data(product_id[k], start_date, end_date, bounding_box, output_folder[k])

