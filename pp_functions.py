import rioxarray as rxr
import matplotlib.pyplot as plt

import geopandas as gpd
import numpy as np
from scipy.signal import convolve2d
import glob
import os
import pandas as pd

import xarray as xr
from datetime import datetime
import rasterio as rio
import netCDF4
import time
from scipy.stats import mode
from scipy.ndimage import median_filter

def fill_nodata(stacked_data):
    t,y,x = stacked_data.shape
    for tt in np.arange(t):
        tt_date = stacked_data[tt,:,:]
        if tt == 0: # For the first time step, any value of 2 is set to 1
            tt_date[tt_date == 2] = 1
        else: # For following timesteps, any value of 2 is set to the value of the previous timestep
            tt_ref = stacked_data[tt-1,:,:]
            tt_date[tt_date == 2] = tt_ref[tt_date == 2] 
        stacked_data[tt,:,:] = tt_date
    return stacked_data

# Calculate the snow cover percentage
def calculate_snow_cover_percentage(stacked_data):
    # Count snow (represented as 1) while ignoring NaNs
    snow_count = np.nansum(stacked_data == 1, axis=(1, 2)) # Axis = (1,2) sums over the y and x dimensions
    # Count total pixels (not NaN) in each time slice
    total_count = np.sum(~np.isnan(stacked_data), axis=(1, 2))

    snow_cover_percentage = (snow_count / total_count) * 100
    return snow_cover_percentage

def create_circular_kernel(radius):
    """Create a circular kernel of specified radius."""
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = radius

    for x in range(size):
        for y in range(size):
            if np.sqrt((x - center) ** 2 + (y - center) ** 2) <= radius:
                kernel[x, y] = 1

    return kernel



def masked_convolution(stacked_data, mask, kernel, threshold_percent):
    time, x,y = stacked_data.shape
    convolved_stack = np.zeros_like(stacked_data)
    for t in range(time):

        # Ensure the mask is boolean: valid_mask is True where canopy is present (mask == 0)
        valid_mask = (mask == 0)

        # Perform convolution on the entire image
        convolved = convolve2d(stacked_data[t], kernel, mode='same', boundary='symm')
    
        # Calculate the threshold based on the percentage of the kernel size
        threshold = int(threshold_percent * np.sum(kernel==1))

        # Create a new binary image based on the neighbor count
        adjusted_snow_image = np.where(convolved >= threshold, 1, 0)

        # Create an output image initialized with original image
        convolved_stack[t] = np.copy(stacked_data[t])
        combined_mask = (valid_mask)&(stacked_data[t]==0)&(adjusted_snow_image==1)
        # Only keep the adjusted snow values where the mask is valid
        convolved_stack[t][combined_mask] = 1

    return convolved_stack





# # use basin_mask to mask the data
# def basin_masking(arr, basin_mask, pixel_buffer):
#     n_pixels = pixel_buffer # change this to match whatever buffer your image has
#     t, y, x = arr.shape
#     cropped_stack = np.zeros((t, y - 2 * n_pixels, x - 2 * n_pixels))  # Initialize the cropped stack
#     for t in range(t):
#         arr[t] = np.where(basin_mask == 1, arr[t], np.nan) 
#         cropped_stack[t] = arr[t, n_pixels:-n_pixels, n_pixels:-n_pixels]
#     return cropped_stack

def calculate_pixel_buffer(buff_raster_path, smaller_raster_path):
    # Open the rasters
    with rio.open(buff_raster_path) as buff_src:
        buff_transform = buff_src.transform
        buff_bounds = buff_src.bounds
        buff_width = buff_src.width
        buff_height = buff_src.height

    with rio.open(smaller_raster_path) as smaller_src:
        smaller_transform = smaller_src.transform
        smaller_bounds = smaller_src.bounds
        smaller_width = smaller_src.width
        smaller_height = smaller_src.height
        # Print the bounding boxes for both rasters
    print("Buffered Raster Bounds:", buff_bounds)
    print("Smaller Raster Bounds:", smaller_bounds)

    # Calculate the differences in the spatial extent
    buffer_left = buff_bounds[0] - smaller_bounds[0]
    buffer_bottom = buff_bounds[1] - smaller_bounds[1]
    buffer_right = buff_bounds[2] - smaller_bounds[2]
    buffer_top = buff_bounds[3] - smaller_bounds[3]

    # Convert the differences to pixel units
    buffer_left_px = int(round(abs(buffer_left / buff_transform[0])))  # Pixel width from the affine transform
    buffer_bottom_px = int(round(abs(buffer_bottom / buff_transform[4])))  # Pixel height from the affine transform
    buffer_right_px = int(round(abs(buffer_right / buff_transform[0])))  # Pixel width from the affine transform
    buffer_top_px = int(round(abs(buffer_top / buff_transform[4])))  # Pixel height from the affine transform

    return buffer_left_px, buffer_right_px, buffer_top_px, buffer_bottom_px


def basin_masking(arr, basin_mask, buff_raster_path, smaller_raster_path):
    buffer_left, buffer_right, buffer_top, buffer_bottom = calculate_pixel_buffer(
        buff_raster_path, smaller_raster_path
    )
    t, y, x = arr.shape

    new_y = y - buffer_top - buffer_bottom
    new_x = x - buffer_left - buffer_right
    assert new_y > 0 and new_x > 0, "Cropping would result in negative dimensions."

    cropped_stack = np.zeros((t, new_y, new_x), dtype=arr.dtype)

    for i in range(t):
        masked = np.where(basin_mask == 1, arr[i], np.nan)
        cropped_stack[i] = masked[buffer_top:y - buffer_bottom, buffer_left:x - buffer_right]

    return cropped_stack

# def basin_masking(arr, basin_mask, buffer_left, buffer_right, buffer_top, buffer_bottom):
#     t, y, x = arr.shape
    
#     # Apply mask first
#     for i in range(t):
#         arr[i] = np.where(basin_mask == 1, arr[i], np.nan)
    
#     # Crop each side based on asymmetric buffers
#     cropped_stack = arr[
#         :,  # all time steps
#         buffer_top : y - buffer_bottom,
#         buffer_left : x - buffer_right
#     ]
    
#     return cropped_stack

def temporal_median_filter(arr, mask, code, window_size):

    assert window_size % 2 == 1, "window_size must be odd"
    # Apply median filter across the full stack
    filtered = median_filter(arr, size=(window_size, 1, 1), mode='nearest')

    valid_mask = np.broadcast_to((mask == code), arr.shape)
    # Replace only where mask is True (1)
    result = arr.copy()
    result[valid_mask] = filtered[valid_mask]

    return result



# find the last dates with consecute snow-present observations
def find_last_consecutive_ones_after_zero(arr):
    max_consecutive_indices = []
    n = len(arr)
    i = 0
    while i < n - 1:
        if arr[i] == 0 and arr[i + 1] == 1:
            start_index = i + 1
            while start_index < n - 1 and arr[start_index] == 1 and arr[start_index + 1] == 1:
                start_index += 1
            consecutive_indices = [start_index]
            if len(consecutive_indices) > len(max_consecutive_indices):
                max_consecutive_indices = consecutive_indices
            i = start_index + 1
        else:
            i += 1
    return max_consecutive_indices




def qaqc_spatial_temp(stacked_data,chm_mask,radius,threshold_percent, basin_mask, buff_raster_fn, small_raster_fn): #, basin_mask):

    stacked_data = masked_convolution(stacked_data, chm_mask[0], create_circular_kernel(radius), threshold_percent)
    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 1, window_size=3)
    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 0, window_size=5) 

    # stacked_data = basin_masking(stacked_data,basin_mask,pixel_buffer)
    # Pass them to your masking function
    stacked_data = basin_masking(stacked_data, basin_mask, buff_raster_fn, small_raster_fn)

    _,y,x = stacked_data.shape
    # print('final data checks ...')
    # for j in range(y):
    #     for i in range(x):
    #         location_3 = find_last_consecutive_ones_after_zero(stacked_data[:,j,i])
    #         if len(location_3) > 0:
    #             stacked_data[:location_3[0],j,i] = 1
    return stacked_data


def qaqc_spatial(stacked_data,chm_mask,radius,threshold_percent, basin_mask, buff_raster_fn, small_raster_fn): #, basin_mask):

    stacked_data = masked_convolution(stacked_data, chm_mask[0], create_circular_kernel(radius), threshold_percent)

    # stacked_data = basin_masking(stacked_data,basin_mask,pixel_buffer)
    stacked_data = basin_masking(stacked_data, basin_mask, buff_raster_fn, small_raster_fn)
    _,y,x = stacked_data.shape
    print('final data checks ...')
    # for j in range(y):
    #     for i in range(x):
    #         location_3 = find_last_consecutive_ones_after_zero(stacked_data[:,j,i])
    #         if len(location_3) > 0:
    #             stacked_data[:location_3[0],j,i] = 1
    return stacked_data


def qaqc_temp(stacked_data, chm_mask,basin_mask, buff_raster_fn, small_raster_fn): #, basin_mask):


    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 1, window_size=3)
    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 0, window_size=5) 
    # stacked_data = basin_masking(stacked_data,basin_mask, pixel_buffer)
    stacked_data = basin_masking(stacked_data, basin_mask, buff_raster_fn, small_raster_fn)
    _,y,x = stacked_data.shape
    print('final data checks ...')
    # for j in range(y):
    #     for i in range(x):
    #         location_3 = find_last_consecutive_ones_after_zero(stacked_data[:,j,i])
    #         if len(location_3) > 0:
    #             stacked_data[:location_3[0],j,i] = 1
    return stacked_data

# Added 1/17
def fill_DSDdates(daydiff,dsd,nodata_ref):
    # determine the actual day since October 1 the observation came from
    dsd_date = daydiff[dsd].astype(float)
    
    temp = np.empty(dsd.shape)
    # loop through the dsd array
    # determine the snow classification on the dsd date
    for j in range(dsd.shape[0]):
        for i in range(dsd.shape[1]):
            time_index = dsd[j,i]
            temp[j,i] = nodata_ref[time_index,j,i]
            

            
    # determine where when dsd occurred on a bad observation pixel account for that in the uncertainty
    filled_y,filled_x = np.where(temp == 2)
    if len(filled_y) > 0:
        for j,i in zip(filled_y,filled_x):
            time_index = dsd[j,i]
            temp = nodata_ref[time_index,j,i]
            while temp == 2:
                time_index = dsd[j,i] - 1
                temp = nodata_ref[time_index,j,i]
            # fill the dsd with the last good date
            dsd[j,i] = time_index
    
    # determine the snow-present day just before that
    temp = daydiff[dsd-1].astype(float)
    # and calculate the uncertainty
    temp[dsd < 0] = np.nan
    dsd_uncertain = dsd_date-temp
    # set the dsd as the halfway point
    dsd_date = np.ceil((dsd_date+temp)/2)#.astype(int)
    return dsd_date,dsd_uncertain