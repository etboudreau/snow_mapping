import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import rioxarray
from datetime import datetime
import glob
import os
import contextily as ctx
import xarray
from rioxarray.exceptions import NoDataInBounds



#### functions
def calc_rgb(ds):
    # Selecting RGB bands
    blue_band = ds.isel(band=0)
    green_band = ds.isel(band=1)
    red_band = ds.isel(band=2)
    nir_band = ds.isel(band=3)
    
    # normalize to help visual understanding
    maxval = green_band.max().values
    minval = green_band.min().values
    red_norm = (red_band - minval) / (maxval - minval)
    green_norm = (green_band - minval) / (maxval - minval)
    blue_norm = (blue_band - minval) / (maxval - minval)
    green_norm = green_norm.where(red_norm <= 1,1)
    blue_norm = blue_norm.where(red_norm <= 1,1)
    red_norm = red_norm.where(red_norm <= 1,1)

    # port to numpy
    red_band = red_band.values
    green_band = green_band.values
    blue_band = blue_band.values
    nir_band = nir_band.values
    
    # Stack normalized bands to create RGB image
    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)
    return red_band,green_band,blue_band,nir_band,rgb_image

def dem_info(DEM,basin):
    dem = rioxarray.open_rasterio(DEM, all_touched=False,drop=True, masked=True).rio.reproject('EPSG:32611', inplace=True).rio.clip(basin.geometry.values, all_touched = False,  drop=True)
    dem.values = np.where(dem.values == dem.rio.nodata, np.nan, dem.values)
    dem.rio.write_nodata(np.nan, inplace=True)
    dem_mean = np.nanmean(dem.values)
    dem_max = np.nanmax(dem.values)
    dem_min = np.nanmin(dem.values)
    return dem, dem_mean, dem_max, dem_min

def create_binary_chm(chm,basin):
    chm_mask = rioxarray.open_rasterio(chm, all_touched=False,drop=True, masked=True).rio.clip(basin.geometry.values, all_touched = False,  drop=True)
    mean = np.nanmean(chm_mask.values)
    max = np.nanmax(chm_mask.values)
    chm_mask.values = np.where(chm_mask<2,1, chm_mask) # OPEN
    chm_mask.values = np.where(chm_mask>=2, 0, chm_mask) # FOREST
    one_cnt = (chm_mask.values == 1).sum()
    zero_cnt = (chm_mask.values == 0).sum()
    fcan= zero_cnt/(one_cnt+zero_cnt)*100
    return chm_mask, mean, max, fcan


# Extracting the dates from the ASO and PS SCA files
# Use this for plotting, comparing the model predictions, or using the function def closest_date():
def extract_dates(aso_tif, ps_sca_tif):
    aso_dates = []
    ps_dates = []
    for aso_file in aso_tif:
        filename = os.path.basename(aso_file)
        date_str = filename.split('_')[2]
        year = date_str[:4]
        month = date_str[4:7]
        day = date_str[7:]
        if "-" in day:
            day = day.split("-")[0]
        date_obj = datetime.strptime(f"{year} {month} {day}", "%Y %b %d")
        aso_dates.append((date_obj, aso_file))
        
    for ps_file in ps_sca_tif:
        filename = os.path.basename(ps_file)
        date_str = filename.split('_')[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        ps_dates.append((date_obj, ps_file))
    return aso_dates, ps_dates

# Hopefully improved efficiency from the function below 
def create_ps_df(ps_sca_tif, basin, chm_mask, name):
    # Load RGI and WBD masks once
    rgi_path = '/home/etboud/projects/data/RGI/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp'
    wbd_path = '/home/etboud/projects/data/masks/NHDWaterbody.shp'
    
    rgi_mask = gpd.read_file(rgi_path).to_crs('EPSG:32611')
    wbd_mask = gpd.read_file(wbd_path).to_crs('EPSG:32611')
    
    # Prepare to collect results
    ps_sca_df = []
    
    for file in ps_sca_tif:
        filename = os.path.basename(file)
        date = filename.split('.')[0].split('_')[0]
        
        # Open SCA raster
        ps_sca = rioxarray.open_rasterio(file, masked=True)
        raster_crs = ps_sca.rio.crs
        
        # Clip to basin geometry
        basin_clip = basin.to_crs(raster_crs).geometry.values
        ps_sca = ps_sca.rio.clip(basin_clip, crs=basin.crs, drop=True)
        ps_sca.values = np.where(np.isnan(ps_sca), 0, ps_sca)
        
        # Clip with RGI and WBD
        ps_sca = ps_sca.rio.clip(rgi_mask.geometry.values, invert=True)
        ps_sca = ps_sca.rio.clip(wbd_mask.geometry.values, invert=True)

        # Count occurrences
        counts = np.unique(ps_sca.values, return_counts=True)
        count_dict = dict(zip(counts[0], counts[1]))
        
        zero_cnt = count_dict.get(0, 0)
        one_cnt = count_dict.get(1, 0)
        two_cnt = count_dict.get(2, 0)
        total_cnt = zero_cnt + one_cnt + two_cnt
        
        one_pct = one_cnt / total_cnt if total_cnt else 0
        zero_pct = zero_cnt / total_cnt if total_cnt else 0
        two_pct = two_cnt / total_cnt if total_cnt else 0

        # Mask out CHM
        # ps_sca_chm = ps_sca.values * chm_mask.values
        # chm_counts = np.unique(ps_sca_chm, return_counts=True)
        # chm_count_dict = dict(zip(chm_counts[0], chm_counts[1]))
        
        # chm_five_cnt = chm_count_dict.get(0, 0)
        # chm_one_cnt = chm_count_dict.get(1, 0)
        # chm_two_cnt = chm_count_dict.get(2, 0)
        # chm_total_cnt = chm_five_cnt + chm_one_cnt + chm_two_cnt
        
        # chm_one_pct = chm_one_cnt / chm_total_cnt if chm_total_cnt else 0
        # chm_five_pct = chm_five_cnt / chm_total_cnt if chm_total_cnt else 0
        # chm_two_pct = chm_two_cnt / chm_total_cnt if chm_total_cnt else 0
        
        # Append results
        ps_sca_df.append({
            "fn": filename.split('.')[0],
            "date": date,
            "basin": name,
            "snow": one_cnt,
            "unobs": two_cnt,
            "nosnow": zero_cnt,
            "fsca_obs": one_pct,
            "f_unobs": two_pct,
            "f_nosnow": zero_pct,
            # "chm_snow": chm_one_cnt,
            # "chm_unobs": chm_two_cnt,
            # "chm_nosnow": chm_five_cnt,
            # "chm_fsca_obs": chm_one_pct,
            # "chm_f_unobs": chm_two_pct,
            # "chm_f_nosnow": chm_five_pct,
            # "PS_check": one_pct + five_pct + two_pct,
            # "chm_check": chm_one_pct + chm_five_pct + chm_two_pct
        })
    
    # Create DataFrame and filter
    ps_sca_year = pd.DataFrame(ps_sca_df)
    ps_sca_year = ps_sca_year[ps_sca_year['f_unobs'] < 0.02]
    ps_sca_year = ps_sca_year.sort_values("date").reset_index(drop=True)
    ps_sca_year['date'] = pd.to_datetime(ps_sca_year['date'], format='%Y%m%d')
    
    return ps_sca_year



def create_aso_df(aso_tif, basin, name, chm_mask):
    # ASO dataframe
    aso_sca_df = []

    for file in aso_tif:

        #extracting filename & date
        filename = os.path.basename(file)
        date_str = filename.split('_')[2]
        # Split the string to extract year, month, and days
        year = date_str[:4]
        month = date_str[4:7]
        day = date_str[7:]
        if "-" in day:
            day = day.split("-")[0]

        date_obj = datetime.strptime(f"{year} {month} {day}", "%Y %b %d")
        date = date_obj.strftime("%Y-%m-%d")



        # Open ASO
        rio_aso = rioxarray.open_rasterio(file, mask_and_scale=True,crs = 'EPSG:32611')
        rio_aso = rio_aso.rio.clip(basin.geometry.values, crs='EPSG:32611', drop=True)
        rio_proj = rio_aso.rio.reproject('EPSG:32611', nodata=np.nan)

        rio_proj.values=np.where(np.isnan(rio_proj), 2, rio_proj.values)
        rio_proj.values = np.where(rio_proj.values>0.1, 1, rio_proj.values)
        rio_proj.values = np.where(rio_proj.values<=0.1, 0, rio_proj.values)
        rio_proj = rio_proj.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
        

        rgi = '/home/etboud/projects/data/RGI/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp'
        rgi_mask = gpd.read_file(rgi).to_crs('EPSG:32611')
        wbd = '/home/etboud/projects/data/masks/NHDWaterbody.shp'
        wbd_mask = gpd.read_file(wbd).to_crs('EPSG:32611')
        rio_proj = rio_proj.rio.clip(rgi_mask.geometry,invert=True) #testing
        rio_proj = rio_proj.rio.clip(wbd_mask.geometry,invert=True) #testing
        
        # # Mask out CHM by muliplying mask by 
        # aso_sca_chm = aso_sca_trimmed.values * chm_mask.values
        # # Converting SCA_CHM to xarray
        # aso_sca_chm = xarray.DataArray(aso_sca_chm, dims=aso_sca_trimmed.dims, coords={dim: aso_sca_trimmed.coords[dim] for dim in sca_trimmed.dims})


        # Total pixel count
        #cnt = rio_proj.values.size
        # Snow absence or pixel outside of domain
        zero_cnt = (rio_proj.values == 0).sum()
        # Classified snow presence
        one_cnt = (rio_proj.values == 1).sum()
        # Unobserved, this includes pixels inside of domain extents that weren't covered by PS obs or pixels where image artifacts were present
        two_cnt = (rio_proj.values == 2).sum()
        # Total pixel count
        cnt = zero_cnt + one_cnt + two_cnt
        #snow presence percentage
        one_pct = one_cnt/cnt
        #snow absence percentage
        zero_pct = zero_cnt/cnt
        #unobserved percentage
        two_pct = two_cnt/cnt
        check = one_pct + zero_pct + two_pct

        #chm_mask.values = np.where(chm_mask==2, np.nan, chm_mask)

        # masked
        # Trimming SCA to CHM (off by 1 pixel)
        #aso_sca_trimmed= rio_proj[:, :, :2715] 

        # # Mask out CHM by muliplying mask by 
        # aso_sca_chm = rio_proj.values * chm_mask.values
        # # Converting SCA_CHM to xarray
        # aso_sca_chm = xarray.DataArray(aso_sca_chm, dims=rio_proj.dims, coords={dim: rio_proj.coords[dim] for dim in rio_proj.dims})

        # chm_five_cnt = (aso_sca_chm.values == 0).sum()
        # chm_one_cnt = (aso_sca_chm.values == 1).sum()
        # chm_two_cnt = (aso_sca_chm.values == 2).sum()
        # chm_cnt = chm_five_cnt + chm_one_cnt + chm_two_cnt
        # chm_one_pct = chm_one_cnt/chm_cnt
        # chm_five_pct = chm_five_cnt/chm_cnt
        # chm_two_pct = chm_two_cnt/chm_cnt


        aso_sca_df.append({
        "fn": filename.split('.')[0],
        "basin": name,
        "date": date,
        "snow": one_cnt,
        "unobs": two_cnt,
        "nosnow": zero_cnt,
        "fsca_obs": one_pct,
        "f_unobs": two_pct,
        "f_nosnow": zero_pct,
        # "chm_snow": chm_one_cnt,
        # "chm_unobs": chm_two_cnt,
        # "chm_nosnow": chm_five_cnt,
        # "chm_fsca_obs": chm_one_pct,
        # "chm_f_unobs": chm_two_pct,
        # "chm_f_nosnow": chm_five_pct
    })
        
    # Create SCA pixel count DataFrame for ASO
    aso_sca_year = pd.DataFrame(aso_sca_df)
    aso_sca_year['date'] =  pd.to_datetime(aso_sca_year['date'], format='%Y-%m-%d')
    aso_sca_year = aso_sca_year.sort_values(by='date')
    
    return aso_sca_year



# Function to find the closest date in the PS SCA files to the ASO date
def closest_date(ps_dates, aso_date):
    # Extract dates from tuples
    ps_date_list = [date for date, _ in ps_dates]
    # Sort the ps_dates
    sorted_list = sorted(ps_date_list)
    previous_date = sorted_list[-1]
    
    for date in sorted_list:
        if date >= aso_date:
            if abs((date - aso_date).days) < abs((previous_date - aso_date).days):
                return date
            else:
                return previous_date
        previous_date = date
    
    return sorted_list[-1]

# Gets multiple dates if 2 or more dates are equally close
def closest_dates(ps_dates, aso_date):
    # Extract dates from tuples
    ps_date_list = [date for date, _ in ps_dates]
    # Sort the ps_dates 
    sorted_list = sorted(ps_date_list)
    
    closest_distance = None
    closest_dates = []

    for date in sorted_list:
        distance = abs((date - aso_date).days)
        
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_dates = [date]
        elif distance == closest_distance:
            closest_dates.append(date)

    return closest_dates

# Comparing 2 trained model predictions with a binary ASO map
# ps_dates and aso_dates can be created from the extract_dates function
# model1_files and model2_files are the paths to the model prediction .tif files
def plot_model_comparison(model1_files, model2_files, ps_dates, aso_dates, basin):
    # Mask out the RGI and WBD Testing
    rgi = '/home/etboud/projects/data/RGI/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp'
    rgi_mask = gpd.read_file(rgi).to_crs('EPSG:32611')
    wbd = '/home/etboud/projects/data/masks/NHDWaterbody.shp'
    wbd_mask = gpd.read_file(wbd).to_crs('EPSG:32611')
    
    for aso_date, aso_file in sorted(aso_dates):
        closest = closest_date(ps_dates, aso_date)
        closest_str = closest.strftime('%Y%m%d')
        
        model1_file = None
        model2_file = None
        
        for file1 in model1_files:
            if closest_str in os.path.basename(file1):
                model1_file = file1
                break
        
        for file2 in model2_files:
            if closest_str in os.path.basename(file2):
                model2_file = file2
                break
        
        if model1_file and model2_file:
            m1_sca = rioxarray.open_rasterio(model1_file, all_touched=False, drop=True, masked=True)
            m1_sca.values = np.where(np.isnan(m1_sca.values), 0, m1_sca.values)
            m1_sca = m1_sca.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
            m1_sca = m1_sca.rio.clip(rgi_mask.geometry,invert=True) #testing
            m1_sca = m1_sca.rio.clip(wbd_mask.geometry,invert=True) #testing
            m1_sca = np.squeeze(m1_sca.values)
            
            
            m2_sca = rioxarray.open_rasterio(model2_file, all_touched=False, drop=True, masked=True)
            m2_sca.values = np.where(np.isnan(m2_sca.values), 0, m2_sca.values)
            m2_sca = m2_sca.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
            m2_sca = m2_sca.rio.clip(rgi_mask.geometry,invert=True) #testing
            m2_sca = m2_sca.rio.clip(wbd_mask.geometry,invert=True) #testing
            m2_sca = np.squeeze(m2_sca.values)

            aso_sca = rioxarray.open_rasterio(aso_file, all_touched=False, drop=True, masked=True)
            aso_sca.values = np.where(np.isnan(aso_sca.values), 0, aso_sca.values)
            aso_sca = aso_sca.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
            aso_sca = aso_sca.rio.clip(rgi_mask.geometry,invert=True) #testing
            aso_sca = aso_sca.rio.clip(wbd_mask.geometry,invert=True) #testing
            aso_sca = np.squeeze(aso_sca.values)
            
            fig, ax = plt.subplots(1, 3, figsize=(14, 6))
            ax[0].imshow(m1_sca, vmin=0, vmax=2, interpolation='none')
            ax[0].set_title('V1')
            
            ax[1].imshow(m2_sca, vmin=0, vmax=2, interpolation='none')
            ax[1].set_title('V2')
            
            ax[2].imshow(aso_sca, vmin=0, vmax=2, interpolation='none')
            ax[2].set_title('ASO')
            
            plt.suptitle(f'Snow Map for ASO Date: {aso_date.strftime("%Y-%m-%d")}, Closest Model Date: {closest_str}')
            plt.tight_layout()
        
        
            plt.show()
        else:
            print(f'No matching files found for the closest date {closest_str}')
            


# Create 4 class raster from binary ASO raster and binary CHM raster to validate 4 class model
def validation_tif_4class(aso_tif, chm_tif, basin, output_tif):
    aso_binary = rioxarray.open_rasterio(aso_tif, mask_and_scale=True).rio.clip(basin.geometry.values, crs=basin.crs, drop=True).squeeze()
    chm_binary = chm_tif.squeeze()
    
    assert aso_binary.shape == chm_binary.shape, "ASO and CHM binary rasters must have the same shape"
    assert aso_binary.rio.crs == chm_binary.rio.crs, "ASO and CHM binary rasters must have the same CRS"
    assert aso_binary.rio.transform() == chm_binary.rio.transform(), "ASO and CHM binary rasters must have the same transform"
    
    combined_classes = np.full(aso_binary.shape, np.nan, dtype=np.float32)
    
    combined_classes[(aso_binary == 0) & (chm_binary == 1)] = 0  # no snow, no canopy
    combined_classes[(aso_binary == 1) & (chm_binary == 0)] = 2  # snow, canopy
    combined_classes[(aso_binary == 1) & (chm_binary == 1)] = 3 # snow, no canopy
    combined_classes[(aso_binary == 0) & (chm_binary == 0)] = 1  # no snow, canopy
    
    combined_classes_da = xarray.DataArray(combined_classes, coords = aso_binary.coords, dims = aso_binary.dims)
    # Update the metadata with the original attributes and set nodata value
    combined_classes_da.rio.write_nodata(np.nan, inplace=True)
    # Write the combined data to a new raster file
    combined_classes_da.rio.to_raster(output_tif, driver='GTiff')

# Create binary validation tif
# aso_tif is a directory of aso file paths

def validation_tif_binary(aso_tif, basin, year, name,EPSG):
    for file in aso_tif:
        if year in file:
            filename = file.split('/')[-1].split('.')[0]
            print(f"Processing file: {filename}")
            try:
                aso = rioxarray.open_rasterio(file, mask_and_scale=True)
                aso = aso.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
                aso = aso.rio.reproject(EPSG, nodata=np.nan)
                aso.values = np.where(aso.values > 0.1, 1, aso.values)
                aso.values = np.where(aso.values <= 0.1, 0, aso.values)
                output_path = os.path.join(f'/home/etboud/projects/data/aso/validation/{name}/{filename}_binary.tif')
                aso.rio.to_raster(output_path)
                print(f"Saved binary TIFF to: {output_path}")
            except NoDataInBounds:
                print(f"No data found in bounds for file: {filename}")


# Function to create a dataset with the predictions from PS and observations from binary ASO map
# Use this to create dataframe for def calculate_metrics(): --- evaluation metrics
# def create_validation_dataset(ps_tif, aso_tif, basin):
    
#     # Mask out the RGI and WBD Testing
#     rgi = '/home/etboud/projects/data/RGI/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp'
#     rgi_mask = gpd.read_file(rgi).to_crs('EPSG:32611')
#     wbd = '/home/etboud/projects/data/masks/NHDWaterbody.shp'
#     wbd_mask = gpd.read_file(wbd).to_crs('EPSG:32611')
    
#     # Open and clip the PS raster dataset
#     with rioxarray.open_rasterio(ps_tif, masked=True) as ds:
        
#         ds.values = np.where(np.isnan(ds.values), 0, ds.values)
#         ds_clipped = ds.rio.clip(basin.geometry)
#         ds_clipped = ds_clipped.rio.clip(rgi_mask.geometry,invert=True) # testing
#         ds_clipped = ds_clipped.rio.clip(wbd_mask.geometry,invert=True) # testing
#         
#         # Initialize ps_points list
#         ps_points = []

#         # Get dimensions of the image
#         height, width = ds_clipped.shape[1], ds_clipped.shape[2]

#         # Iterate over all pixel coordinates/indices
#         for y in range(height):
#             for x in range(width):
#                 ps_points.append((x, y, ds_clipped.values[0, y, x]))

#     # Create dataframe for PS data
#     ps_df = pd.DataFrame(ps_points, columns=['x', 'y', 'predict'])

#     # Open and clip the ASO raster dataset
#     with rioxarray.open_rasterio(aso_tif, masked=True) as ds:
#         ds.values = np.where(np.isnan(ds.values), 0, ds.values)
#         ds_clipped = ds.rio.clip(basin.geometry, drop=False)
#         ds_clipped = ds_clipped.rio.clip(rgi_mask.geometry,invert=True) #tetsing
#         ds_clipped = ds_clipped.rio.clip(wbd_mask.geometry,invert=True) #tetsing
#         
#         # Initialize aso_points list
#         aso_points = []

#         # Get dimensions of the clipped image
#         height, width = ds_clipped.shape[1], ds_clipped.shape[2]

#         # Iterate over all pixel coordinates/indices
#         for y in range(height):
#             for x in range(width):
#                 aso_points.append((x, y, ds_clipped.values[0, y, x]))

#     # Create dataframe for aso label
#     aso_df = pd.DataFrame(aso_points, columns=['x', 'y', 'obs'])
    
#     # Merge the two dataframes on the pixel coordinates/indices
#     df = pd.merge(aso_df, ps_df, on=['x', 'y'])
#     return df
# # above chunk is the same as below but does not include plotting

# Function to create a dataset with the predictions from PS and observations from binary ASO map
# Use this to create dataframe for def calculate_metrics(): --- evaluation metrics
def create_validation_dataset(ps_tif, aso_tif, basin):
    # Mask out the RGI and WBD Testing
    rgi = '/home/etboud/projects/data/RGI/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp'
    rgi_mask = gpd.read_file(rgi).to_crs('EPSG:32611')
    wbd = '/home/etboud/projects/data/masks/NHDWaterbody.shp'
    wbd_mask = gpd.read_file(wbd).to_crs('EPSG:32611')
    
    # Open and clip the PS raster dataset
    with rioxarray.open_rasterio(ps_tif, masked=True) as ds:
        ds.values = np.where(np.isnan(ds.values), 0, ds.values)
        ds_clipped = ds.rio.clip(basin.geometry)
        ds_clipped = ds_clipped.rio.clip(rgi_mask.geometry, invert=True)
        ds_clipped = ds_clipped.rio.clip(wbd_mask.geometry, invert=True)

        # Plot the clipped data
        plt.figure(figsize=(10, 8))
        plt.imshow(ds_clipped.isel(band=0), cmap='viridis')  # Change band if necessary
        plt.colorbar(label='PS Values')
        plt.title('Clipped PS Raster')
        plt.show()

        # Initialize ps_points list
        ps_points = []
        height, width = ds_clipped.shape[1], ds_clipped.shape[2]

        # Iterate over pixel coordinates
        for y in range(height):
            for x in range(width):
                ps_points.append((x, y, ds_clipped.values[0, y, x]))

    # Create dataframe for PS data
    ps_df = pd.DataFrame(ps_points, columns=['x', 'y', 'predict'])

    # Open and clip the ASO raster dataset
    with rioxarray.open_rasterio(aso_tif, masked=True) as ds:
        ds.values = np.where(np.isnan(ds.values), 0, ds.values)
        ds_clipped = ds.rio.clip(basin.geometry, drop=False)
        ds_clipped = ds_clipped.rio.clip(rgi_mask.geometry, invert=True)
        ds_clipped = ds_clipped.rio.clip(wbd_mask.geometry, invert=True)

        # Plot the clipped data
        plt.figure(figsize=(10, 8))
        plt.imshow(ds_clipped.isel(band=0), cmap='viridis')  # Change band if necessary
        plt.colorbar(label='ASO Values')
        plt.title('Clipped ASO Raster')
        plt.show()

        # Initialize aso_points list
        aso_points = []
        height, width = ds_clipped.shape[1], ds_clipped.shape[2]

        # Iterate over pixel coordinates
        for y in range(height):
            for x in range(width):
                aso_points.append((x, y, ds_clipped.values[0, y, x]))

    # Create dataframe for ASO data
    aso_df = pd.DataFrame(aso_points, columns=['x', 'y', 'obs'])
    
    # Merge the two dataframes on the pixel coordinates
    df = pd.merge(aso_df, ps_df, on=['x', 'y'])
    return df




# Function to calculate the evaluation metrics
def calculate_metrics(df):
    from sklearn.metrics import cohen_kappa_score
    out = pd.DataFrame(data = {'precision': [0], 
                                    'recall':[0], 
                                    'f1':[0],
                                    'sensitivity':[0],
                                    'specificity':[0],
                                    'balanced_accuracy':[0], 
                                    'accuracy':[0],
                                    'kappa':[0],
                                    'TP':[0],
                                    'TN':[0],
                                    'FP':[0],
                                    'FN':[0]})
    
    # true positive and true negative
    subdf = df[df.predict == df.obs] # true prediction
    TP = len(subdf[subdf.predict == 1].index) # true positive
    TN = len(subdf[subdf.predict == 0].index) # true negative
    

    # false positive and false negative
    subdf = df[df.predict != df.obs] # false prediction 
    FN = len(subdf[subdf.predict == 0].index)# false negative
    FP = len(subdf[subdf.predict == 1].index) # false positive

    
    if (TP + FP) == 0 or (TP + FN) == 0 or (TN + FP) == 0 or (TP+FN) == 0:
        return out
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)

        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        balanced_accuracy = (sensitivity+specificity)/2
        accuracy = (TP+TN)/(TP+TN+FP+FN)

        f1 =  0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        kappa = cohen_kappa_score(df.predict, df.obs)
        out = pd.DataFrame(data = {'precision': [precision], 
                                        'recall':[recall], 
                                        'f1':[f1],
                                        'sensitivity':[sensitivity],
                                        'specificity':[specificity],
                                        'balanced_accuracy':[balanced_accuracy], 
                                        'accuracy':[accuracy],
                                        'kappa':[kappa],
                                        'TP':[TP],
                                        'TN':[TN],
                                        'FP':[FP],
                                        'FN':[FN]})
        return out
    
    