# RoostRingSearch
# ---------------

# Melinda Kleczynski
# University of Delaware

# Katie Bird
# University of Delaware

# Chad Giusti
# Oregon State University

# Jeffrey Buler
# University of Delaware

# MK wrote the code.
# KB ran early versions of the package and provided feedback and suggestions.
# CG and JB provided technical background and project suggestions.


####################


"""
Python 3 package for finding swallow roost rings on US weather surveillance radar

Main functions
--------------
find_roost_rings
morning_exodus

Additional functions
--------------------
empty_roost_df
get_example
get_localtime_str
get_radar_arrays
get_ring_center_coords
get_scan_dts_in_window
get_scan_field_str
get_single_product_array
get_station_latlon
hilite_rings_found
list_season_dates
make_annulus
make_prefix_str
near_sunrise_time
no_files_that_day
prefix_to_scan_pyart
single_filter_sweep

Dependencies
------------
arm-pyart 1.14.1
astral 2.2  
boto3 1.21.43  
botocore 1.24.43 
haversine 2.5.1
matplotlib 3.5.1
numpy 1.22.3
pandas 1.4.2
pytz 2022.1
scikit-image 0.19.2
scipy 1.8.0
timezonefinder 5.2.0
"""


####################


import numpy as np
import numpy.ma as ma

import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import convolve
from scipy.interpolate import griddata
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import dilation, disk

import pyart
from pyart.io import nexrad_common

from astral.sun import sun
from astral import LocationInfo

from haversine import inverse_haversine, Direction

import boto3
import botocore  

import datetime
from pytz import timezone
from timezonefinder import TimezoneFinder

import tempfile
import warnings


####################


# main functions


def morning_exodus(station_str, 
                   scan_date, 
                   cutoff_distance = 150,
                   min_reflectivity = 0,
                   max_background_noise = 0.05,
                   min_signal = 0.3, 
                   minute_offset = -30, 
                   minute_duration = 90, 
                   display_output = False):
    
    """
    Check all the scans for the morning of a given date and station. Aggregate the results.

    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    scan_date : [year (int), month (int), day (int)]
        Date of the requested scan
    cutoff_distance : int, optional
        Restrict to x and y within cutoff_distance km from the radar station
        Default: cutoff_distance = 150
    min_reflectivity : float, optional
        Minimum reflectivity threshold
        Default: min_reflectivity = 0       
    max_background_noise : float, optional
        Allowed ratio of positive array values outside the ring region
        Default: max_background_noise = 0.05
    min_signal : float, optional
        Required ratio of positive array values in the ring region
        Default : min_signal = 0.3 
    minute_offset : int, optional
        Minutes before/after sunrise (negative/positive values)
        Default: minute_offset = -30
    minute_duration : int, optional
        Length of the window of time, in minutes
        Default: minute_duration = 90
    display_output : bool, optional
        Whether to create plots and display messages about output
        Default: display_output = False

    Returns
    -------
    latlon_coord_df or empty_roost_df : DataFrame
        Latitude and longitude of centers of potential roost rings
    """
    
    if no_files_that_day(station_str, scan_date):
        
        return empty_roost_df()
    
    starting_dt = near_sunrise_time(station_str, scan_date, minute_offset) 
    valid_scan_dts = get_scan_dts_in_window(station_str, starting_dt, minute_duration)  
    
    all_times_ring_center_pixels = []
    reflectivity_scan_times = []
    
    for scan_dt in valid_scan_dts:

        search_output = find_roost_rings(station_str, 
                                         scan_dt, 
                                         cutoff_distance = cutoff_distance, 
                                         min_reflectivity = min_reflectivity, 
                                         max_background_noise = max_background_noise, 
                                         min_signal = min_signal, 
                                         display_output = display_output)
        
        if search_output["success"]:
            all_times_ring_center_pixels += [search_output["ring center pixels"]]
            reflectivity_scan_times += [search_output["reflectivity scan time"]]

    if len(all_times_ring_center_pixels) > 0:
        
        ring_center_coords = get_ring_center_coords(all_times_ring_center_pixels, station_str, display_output = display_output)
        center_coords_latlon = ring_center_coords["center coords (lat/lon)"]
        
        n_rings_found = len(center_coords_latlon)

        if n_rings_found > 0:
            
            latlon_coord_df = pd.DataFrame(center_coords_latlon, columns = ['latitude', 'longitude'])
            latlon_coord_df['first detection'] = np.array(reflectivity_scan_times)[ring_center_coords["first scan indices"]]
            latlon_coord_df['station name'] = [station_str] * n_rings_found
            latlon_coord_df['year'] = [scan_date[0]] * n_rings_found
            latlon_coord_df['month'] = [scan_date[1]] * n_rings_found
            latlon_coord_df['day'] = [scan_date[2]] * n_rings_found

            return latlon_coord_df
        
        else:
            
            return empty_roost_df()

    else: 
        
        return empty_roost_df()


def find_roost_rings(station_str,
             scan_dt,
             cutoff_distance = 150,
            
             # parameters
             min_reflectivity = 0,     
             max_background_noise = 0.05, 
             min_signal = 0.3,
                     
             # display options
             display_output = False,
             figure_length = 6,             
             filename_suffix = ''
             ):
    
    """ Look for roost rings in a single set of scans
    
    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    scan_dt : datetime
        Date and time of the requested scan
    cutoff_distance : int, optional
        Restrict to x and y within cutoff_distance km from the radar station
        Default: cutoff_distance = 150
    min_reflectivity : float, optional
        Minimum reflectivity threshold
        Default: min_reflectivity = 0       
    max_background_noise : float, optional
        Allowed ratio of positive array values outside the ring region
        Default: max_background_noise = 0.05
    min_signal : float, optional
        Required ratio of positive array values in the ring region
        Default : min_signal = 0.3    
    display_output : bool, optional
        Whether to create plots and display messages about output
        Default: display_output = False
    figure_length : float, optional
        Figure size in inches   
        Default: figure_length = 6   
    filename_suffix : string, optional
        String to add to the end of plot filenames, if applicable
        Default: filename_suffix = ''
     
    Returns
    -------
    * Dictionary with the following keys *
    success : bool
        True if the function ran without any issues, False if an error was encountered
        This function will continue running even when certain errors arise, so that
        large amounts of data can be processed without supervision
    latlon coords : array
        Latitude and longitude of centers of potential roost rings
    reflectivity array : array
        Interpolated reflectivity data for the requested scan
    fill value : float
        Value used for unavailable data in the reflectivity scan
    results mask : array
        Masks out reflectivity not contained in a roost ring
    ring center pixels : array
        Array elements which could correspond to the center of a roost ring
    reflectivity scan time : string
        Date and time of reflectivity scan in format to be easily converted to a posix
    """

    ### get scan info
       
    scan_prefix = make_prefix_str(station_str, scan_dt)
    scan_pyart = prefix_to_scan_pyart(scan_prefix, display_output = display_output)
    if scan_pyart == []:  # there was an error trying to access that scan
        return {"success": False}
    
    localtime_str = get_localtime_str(station_str, scan_dt)
    title_str = station_str + " " + "Radar, " + localtime_str
    
    ### get radar arrays
    
    radar_array_grid, fill_value, radar_bool_allcorr, radar_bool_lowcorr = get_radar_arrays(scan_pyart, cutoff_distance, min_reflectivity)  

    ### look for roost rings
 
    inner_radii = [k for k in range(3, 10)] + [3*k for k in range(4, 9)]
    ring_widths = [k for k in range(3, 10)] + [3*k for k in range(4, 7)]  
    ring_structures = [(inner_radius, ring_width) for inner_radius in inner_radii for ring_width in ring_widths if ring_width/2 <= inner_radius]
        
    sweep_results = np.array([single_filter_sweep(radar_bool_allcorr, radar_bool_lowcorr, max_background_noise, min_signal, 
                                                  ring_structure) for ring_structure in ring_structures]) 

    ring_center_pixels = np.amax(sweep_results[:, 0], axis = 0)
    possible_rings = np.amax(sweep_results[:, 1], axis = 0)
    results_mask = 1 - possible_rings
        
    ring_center_coords = get_ring_center_coords([ring_center_pixels], station_str)  
    
    ### timestamp
    
    radar_scan_display = pyart.graph.RadarDisplay(scan_pyart)
    refl0_scan_summary = radar_scan_display.generate_title('reflectivity', 0)

    refl0_scan_time = refl0_scan_summary.split(" ")[3]
    refl0_scan_time = refl0_scan_time.split(".")[0]
    refl0_scan_time = refl0_scan_time.replace("T", " ")
    
    ### view results

    if display_output:

        scan_field_str = get_scan_field_str(scan_pyart)
        
        hilite_rings_found(title_str, scan_field_str, radar_array_grid, fill_value, results_mask, ring_center_coords["center coords (km)"], figure_length, filename_suffix)
    
    return {"success": True, 
            "latlon coords": ring_center_coords["center coords (lat/lon)"], 
            "reflectivity array": radar_array_grid,
            "fill value": fill_value, 
            "results mask": results_mask, 
            "ring center pixels": ring_center_pixels, 
            "reflectivity scan time": refl0_scan_time}


####################


# get coordinates of ring centers


def get_ring_center_coords(ring_center_pixel_arrays, station_str, display_output = False, figure_length = 6):    
    
    """ We already marked pixels which are likely the center of a roost ring. It is possible that several "center" pixels were identified for a single roost ring. They may not form a single connected component if the shape of that roost ring irregular. So we apply dilation to hopefully form one connected component for the center of each roost ring. We use that to determine a location for each roost ring.
    
    Parameters
    ----------
    ring_center_pixel_arrays : list of arrays
        Array values of 1 indicate that a roost ring could be centered there
    station_str : string
        Four letter identifier for the radar station
    display_output : bool, optional
        Whether to create plots and display messages about output
        Default: display_output = False
    figure_length : float, optional
        Figure size in inches if plotting
        Default: figure_length = 6
        
    Returns
    -------
    * Dictionary with the following keys *
    center coords (index) : array
        Coordinates of centers of potential roost rings, as array indices
    center coords (km) : array
        Coordinates of centers of potential roost rings, in km from radar station
    center coords (lat/lon) : array
        Latitude and longitude of centers of potential roost rings
    first scan indices : array
        First scan where each ring was seen
    """
    
    station_lat, station_lon = get_station_latlon(station_str)
    
    n_arrays = len(ring_center_pixel_arrays)
    array_dim = np.shape(ring_center_pixel_arrays[0])[0]
    cutoff_distance = (array_dim - 1) // 2
    
    min_timestamps = np.zeros((array_dim, array_dim), dtype = np.int64)
    
    # don't use 0 as a timestamp, because it's the background value - so timestamps are offset by 1 from indices
    for i in range(array_dim):
        for j in range(array_dim):
            that_pixel = [(array_iter + 1)*ring_center_pixel_arrays[array_iter][i, j] for array_iter in range(n_arrays)]
            if max(that_pixel) > 0:
                min_timestamp = min([val for val in that_pixel if val > 0])
                min_timestamps[i, j] = min_timestamp
    
    bool_signal_array = sum(ring_center_pixel_arrays) > 0
    
    labeled_centers, max_label = label(dilation(bool_signal_array, disk(3)), return_num = True)   
    
    first_scan_indices = np.zeros(max_label, dtype = np.int64)
    center_coords_index = np.zeros((0, 2))

    for label_iter in range(1, max_label + 1):

        timestamped_center = min_timestamps * (labeled_centers == label_iter)
        possible_timestamps = np.unique(timestamped_center)
        possible_timestamps = [pt for pt in possible_timestamps if pt > 0]
        min_timestamp = min(possible_timestamps)
        min_timestamp_center = (timestamped_center == min_timestamp)
        center_coords_index = np.vstack([center_coords_index, ndimage.center_of_mass(min_timestamp_center)])
        first_scan_indices[label_iter - 1] = min_timestamp - 1
    
    if len(center_coords_index) > 0:
        center_coords_km = np.transpose(np.vstack([center_coords_index[:, 1] - cutoff_distance, cutoff_distance - center_coords_index[:, 0]]))
    else:
        center_coords_km = []
    
    center_coords_latlon = []
    for coord_pair in center_coords_km:
        east_of_station = inverse_haversine((station_lat, station_lon), coord_pair[0], Direction.EAST)
        northeast_of_station = inverse_haversine(east_of_station, coord_pair[1], Direction.NORTH)
        center_coords_latlon += [list(northeast_of_station)]
    center_coords_latlon = np.array(center_coords_latlon)
    
    if display_output and len(center_coords_km) > 0:

        fig, ax = plt.subplots(figsize = (figure_length, figure_length))
        
        ax.scatter(center_coords_km[:, 0] + cutoff_distance, cutoff_distance - center_coords_km[:, 1], c = 'gold', edgecolor = 'k', s = 25, linewidth = 1.75)
        
        original_ticks = [50*i for i in range(1, 2*cutoff_distance//50)]
        new_xticks = [tick - cutoff_distance for tick in original_ticks]
        new_yticks = [cutoff_distance - tick for tick in original_ticks]
        ax.set_xticks(original_ticks, new_xticks)
        ax.set_yticks(original_ticks, new_yticks)
        ax.tick_params(labelsize = 12)
        
        ax.set_xlim(0, len(bool_signal_array))
        ax.set_ylim(len(bool_signal_array), 0)
        
        ax.set_xlabel('Distance (km)', fontsize = 12)
        ax.set_ylabel('Distance (km)', fontsize = 12)
        ax.set_title('Potential Roost Ring Centers', fontsize = 12)
        
        plt.show()
        
    return {"center coords (index)": center_coords_index, 
            "center coords (km)": center_coords_km, 
            "center coords (lat/lon)": center_coords_latlon, 
            "first scan indices": first_scan_indices}


####################


# functions for finding/accessing scans


def get_example(ex_num):  
    
    """ Some example scans with roost rings.
    These were used in developing the package.
    
    Parameters
    ----------
    ex_num : int
        Request example 1 or 2
    
    Returns
    -------
    station_str, [year (int), month (int), day (int)]
    
    """

    if ex_num == 1:
        
        return 'KDOX', [2021, 10, 1]
    
    elif ex_num == 2:
    
        print('example 2 from: https://www.weather.gov/mlb/Doppler_Dual_Pol_Weather_Radar')
        return 'KMLB', [2018, 2, 19]  
    
    else:
        
        print('please choose example 1 or example 2')


def near_sunrise_time(station_str, scan_date, minute_offset):  
    
    """
    Obtain a datetime a given number of minutes from sunrise
    
    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    scan_date : [year (int), month (int), day (int)]
        Date of the requested scan
    minute_offset : int
        Minutes before/after sunrise (negative/positive values)
        
    Returns
    -------
    near_sunrise_dt : datetime
    """
    
    station_lat, station_lon = get_station_latlon(station_str)
    [scan_year, scan_month, scan_day] = scan_date
        
    station_obs = LocationInfo(latitude = station_lat, longitude = station_lon).observer
    
    sunrise_dt = sun(station_obs, datetime.date(scan_year, scan_month, scan_day))['sunrise']
    near_sunrise_dt = sunrise_dt + datetime.timedelta(minutes = minute_offset) 
    
    return near_sunrise_dt


def get_scan_dts_in_window(station_str, starting_dt, minute_duration):  
    
    """
    For a given radar station and window of time, get a list of datetimes corresponding to available scan data
    
    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    starting_scan_dt : datetime
        Time to start looking for scan data
    minute_duration : int
        Length of the window of time, in minutes
        
    Returns
    -------
    valid_scan_dts : list of datetimes
        Datetimes within minute_duration minutes after starting_scan_dt for which there is scan data available at the given station
    """

    my_configuration = botocore.client.Config(signature_version = botocore.UNSIGNED)
    nexrad2_bucket = boto3.resource("s3", config = my_configuration).Bucket("noaa-nexrad-level2")

    valid_scan_dts = []

    for i in range(minute_duration):

        new_dt = starting_dt + datetime.timedelta(minutes = i)
        prefix_str = make_prefix_str(station_str, new_dt)
        prefix_scan_keys = [available_file.key for available_file in nexrad2_bucket.objects.filter(Prefix = prefix_str)]
        if len(prefix_scan_keys) > 0:
            valid_scan_dts += [new_dt]

    return valid_scan_dts


def list_season_dates(scan_year):
    
    """
    List dates from June 15 to September 15 (inclusive)

    Parameters
    ----------
    scan_year : int
        Year to use in scan dates

    Returns
    -------
    season_dates : list
        List of dates, each in the format [year (int), month (int), day (int)]
    """

    # June 15 to September 15 (inclusive)
    June_dates = [[scan_year, 6, date_iter] for date_iter in range(15, 31)]
    July_dates = [[scan_year, 7, date_iter] for date_iter in range(1, 32)]
    August_dates = [[scan_year, 8, date_iter] for date_iter in range(1, 32)]
    September_dates = [[scan_year, 9, date_iter] for date_iter in range(1, 16)]

    season_dates = June_dates + July_dates + August_dates + September_dates
    
    return season_dates


def make_prefix_str(station_str, scan_dt):  
    
    """
    Get a file prefix for the given station and datetime
    
    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    scan_dt : datetime
        Date and time of the requested scan
        
    Returns
    -------
    prefix_str : string
        File prefix for the given scan
    """
    
    year_str = str(scan_dt.year)
    month_str = str(scan_dt.month)
    day_str = str(scan_dt.day)
    hour_str = str(scan_dt.hour)
    minute_str = str(scan_dt.minute)
    
    if len(month_str) == 1:
        month_str = '0' + month_str
        
    if len(day_str) == 1:
        day_str = '0' + day_str
        
    if len(hour_str) == 1:
        hour_str = '0' + hour_str    
        
    if len(minute_str) == 1:
        minute_str = '0' + minute_str
        
    path_str = year_str + "/" + month_str + "/" + day_str + "/" + station_str + "/"
    file_str = station_str + year_str + month_str + day_str + "_" + hour_str + minute_str
    
    prefix_str = path_str + file_str
    
    return prefix_str


def prefix_to_scan_pyart(prefix_str, display_output = False):    
    
    """
    Given a file prefix, return a pyart scan object.
    Data pulled from the noaa-nexrad-level2 bucket on AWS; see https://registry.opendata.aws/noaa-nexrad/
    
    Parameters
    ----------
    prefix_str : string
        File prefix for the given scan
    display_output : bool, optional
        Whether to create plots and display messages about output
        Default: display_output = False
        
    Returns
    -------
    scan_pyart : pyart scan object
    """
    
    my_configuration = botocore.client.Config(signature_version = botocore.UNSIGNED)
    nexrad2_bucket = boto3.resource("s3", config = my_configuration).Bucket("noaa-nexrad-level2")
    
    station_day_scan_keys = [available_file.key for available_file in nexrad2_bucket.objects.filter(Prefix = prefix_str)]
    station_day_scan_keys = [sds_key for sds_key in station_day_scan_keys if "MDM" not in sds_key]  # remove MDM files
    scan_key = station_day_scan_keys[0]
    
    scan_object = nexrad2_bucket.Object(scan_key)
    
    with tempfile.TemporaryDirectory() as temp_dir_name:  
        temp_nexrad_file = temp_dir_name + "/temp_nexrad_file"  

        with open(temp_nexrad_file, "wb") as data:
            scan_object.download_fileobj(data)
            
        # want the code to keep running and move on to the next scan if there's a pyart error
        try:
            with warnings.catch_warnings(record = True) as pyart_warning:
                
                scan_pyart = pyart.io.read_nexrad_archive(temp_nexrad_file)  
                
                if len(pyart_warning) > 0:
                    warning_message = str(pyart_warning[0].message)
                    if "fixed angle data will be missing" in warning_message:
                        print("Warning: Fixed angle data missing")
                
        except (OSError, IndexError, ValueError) as pyart_error:
            if display_output:
                print('pyart error:', pyart_error, '- no analysis for:', prefix_str)
            return []
        
    return scan_pyart


def get_scan_field_str(scan_pyart):
      
    """
    Return a reflectivity label for use in plots, etc

    Parameters
    ----------
    scan_pyart : pyart scan object

    Returns
    -------
    scan_field_str : string
        Nicely formatted reflectivity label including scan angle
    """
    
    scan_field_str = scan_pyart.fields['reflectivity']['long_name']
    
    scan_angle = scan_pyart.fixed_angle['data'][0]
    
    if scan_angle > 0:
        scan_angle_units = scan_pyart.fixed_angle['units']
        scan_angle_str = ' (%1.1f %s)'%(scan_angle, scan_angle_units)
        scan_angle_str = scan_angle_str.replace(' d', ' D')
        scan_field_str = scan_field_str + scan_angle_str
    
    return scan_field_str

    
####################


# functions for getting radar arrays


def get_single_product_array(scan_pyart, cutoff_distance, field_name):  

    """
    Extract an array of data for a selected radar product from a pyart scan object

    Parameters
    ----------
    scan_pyart : pyart scan object
    cutoff_distance : int, optional
        Restrict to x and y within cutoff_distance km from the radar station
    field_name : string
        'reflectivity', 'clutter_filter_power_removed', or 'cross_correlation_ratio'

    Returns
    -------
    grid_radar : array
        sweep 0 data for selected radar product
    fill_value : float
        Value used for unavailable data in the scan
    """
    
    ### data
    sweep_num = 0
    masked_polar = scan_pyart.get_field(sweep_num, field_name)
    gate_x, gate_y, gate_z = scan_pyart.get_gate_x_y_z(sweep_num)

    ### m to km
    gate_x /= 1000
    gate_y /= 1000
    # won't use gate_z

    ### fill in masked values
    fill_value = scan_pyart.fields[field_name]['_FillValue']
    filled_polar = masked_polar.filled(fill_value)

    ### flatten arrays
    flat_length = np.shape(gate_x)[0]*np.shape(gate_x)[1]
    gate_x_flat = gate_x.reshape((flat_length,))
    gate_y_flat = gate_y.reshape((flat_length,))
    filled_polar_flat = filled_polar.reshape((flat_length,))

    ### new x and y values
    grid_x, grid_y = np.mgrid[-cutoff_distance:cutoff_distance + 1, -cutoff_distance:cutoff_distance + 1]

    ### interpolate
    grid_radar = griddata((gate_x_flat, gate_y_flat), filled_polar_flat, (grid_x, grid_y), method = 'nearest')

    ### reorient
    grid_radar = np.transpose(grid_radar)
    grid_radar = grid_radar[::-1]
    
    return grid_radar, fill_value


def get_radar_arrays(scan_pyart, cutoff_distance, min_reflectivity):
    
    """
    Extract and process arrays from pyart scan object

    Parameters
    ----------
    scan_pyart : pyart scan object
    cutoff_distance : int
        Restrict to x and y within cutoff_distance km from the radar station
    min_reflectivity : float
        Minimum reflectivity threshold 

    Returns
    -------
    grid_refl : array
        Interpolated reflectivity data
    fill_refl : float
        Value used for unavailable data in the reflectivity scan
    radar_bool_allcorr : array
        processed reflectivity array
    radar_bool_lowcorr : array
        processed reflectivity array with precipitation removed
    """    
    
    ### get reflectivity
    
    grid_refl, fill_refl = get_single_product_array(scan_pyart, cutoff_distance, 'reflectivity')
    mask = grid_refl == fill_refl  
    
    ### remove clutter
    
    if 'clutter_filter_power_removed' in scan_pyart.fields.keys():
    
        grid_clutter, fill_clutter = get_single_product_array(scan_pyart, cutoff_distance, 'clutter_filter_power_removed')
        mask = np.logical_or(mask, grid_clutter != fill_clutter)  
        
    ### threshold
        
    radar_bool_allcorr = ma.array(grid_refl, mask = mask).filled(-100) > min_reflectivity   
    
    ### remove precipitation
    
    if 'cross_correlation_ratio' in scan_pyart.fields.keys():  

        grid_corr, fill_corr = get_single_product_array(scan_pyart, cutoff_distance, 'cross_correlation_ratio')
        
        high_corr = grid_corr > 0.95  # Dokter et al 2018 (bioRad) recommends 0.95
        med_high_corr = ndimage.median_filter(high_corr, footprint = disk(2))  
        dil_high_corr = dilation(med_high_corr, disk(10))  
        corr_mask = np.maximum(dil_high_corr, high_corr)
        
        mask = np.logical_or(mask, corr_mask)
        radar_bool_lowcorr = ma.array(grid_refl, mask = mask).filled(-100) > min_reflectivity   

    else:
        
        radar_bool_lowcorr = copy(radar_bool_allcorr)
    
    return grid_refl, fill_refl, radar_bool_allcorr, radar_bool_lowcorr


####################


# helper functions to search an array for roost rings


def make_annulus(large_rad, small_rad, filter_dim):
    
    """
    Construct an annulus with the requested structure

    Parameters
    ----------
    large_rad : int
        outer radius of the annulus
    small_rad : int
        inner radius of the annulus
    filter_dim : odd integer
        returned array will have shape (filter_dim, filter_dim)

    Returns
    -------
    annulus : array
        square array with 1s in the shape of an annulus
    """
    
    large_disk = disk(large_rad, dtype = int)
    large_disk = np.pad(large_disk, (filter_dim - len(large_disk)) // 2)
    if small_rad == 0:
        return large_disk
    
    small_disk = disk(small_rad, dtype = int)
    small_disk = np.pad(small_disk, (filter_dim - len(small_disk)) // 2)
    annulus = large_disk - small_disk
    return annulus


def single_filter_sweep(radar_bool_allcorr, radar_bool_lowcorr, max_background_noise, min_signal, ring_structure):   
    
    """ Scan over the array. 
    Check for sufficient signal in the positive filter region and not too much signal in the negative filter regions. 
    If both conditions are met, probably a roost ring there. 
    
    Parameters
    ----------
    radar_bool_allcorr : array
        processed reflectivity array
    radar_bool_lowcorr : array
        processed reflectivity array with precipitation removed
    max_background_noise : float
        Allowed ratio of positive array values outside the ring region
    min_signal : float
        Required ratio of positive array values in the ring region
    ring_structure : tuple
        inner_radius, ring_width of annulus to look for

    Returns
    -------
    [ring_center_pixels, possible_rings] : [array, array]
        Ring centers, rings found
    """
    
    inner_radius, ring_width = ring_structure
    
    ### get filters
    
    outer_penalty_width = 2
    outer_buffer_width = 1
    
    center_distance = inner_radius + ring_width + outer_buffer_width + outer_penalty_width
    filter_dim = 2 * center_distance + 1
    
    positive_filter = make_annulus(inner_radius + ring_width, inner_radius, filter_dim)
    negative_filter_out = make_annulus(center_distance, center_distance - outer_penalty_width, filter_dim)
    negative_filter_in = make_annulus(np.ceil(inner_radius / 2), 0, filter_dim)
    
    ### compute amount of noise
    
    inside_noise_ratio = np.pad(convolve(radar_bool_allcorr, negative_filter_in, mode = 'valid'), center_distance) / np.sum(negative_filter_in)
    outside_noise_ratio = np.pad(convolve(radar_bool_allcorr, negative_filter_out, mode = 'valid'), center_distance) / np.sum(negative_filter_out)
    background_noise_ratio = np.maximum(outside_noise_ratio, inside_noise_ratio)
    
    ### compute amount of signal
    
    positive_filter_sum = np.sum(positive_filter)
    signal_ratio = np.pad(convolve(radar_bool_lowcorr, positive_filter, mode = 'valid'), center_distance) / positive_filter_sum
    
    ### check suitability and make output arrays

    condition1 = background_noise_ratio <= max_background_noise   
    condition2 = signal_ratio >= min_signal
    all_conditions = np.logical_and(condition1, condition2)    
    
    ring_center_pixels = 1 * all_conditions
    possible_rings = np.zeros(np.shape(radar_bool_allcorr))
        
    center_is, center_js = np.nonzero(ring_center_pixels)
    for (center_i, center_j) in zip(center_is, center_js):
        
        i = center_i - center_distance
        j = center_j - center_distance
        
        in_ring_signal = np.multiply(positive_filter, radar_bool_allcorr[i:i+filter_dim, j:j+filter_dim])
        possible_rings[i:i+filter_dim, j:j+filter_dim] = np.maximum(possible_rings[i:i+filter_dim, j:j+filter_dim], in_ring_signal)  
   
    return [ring_center_pixels, possible_rings]


####################


# displaying results


def hilite_rings_found(title_str, scan_field_str, radar_array_grid, fill_value, results_mask, center_coords, figure_length, filename_suffix):  
    
    """ Plot the roost rings we found and their centers. 
    
    Parameters
    ----------
    title_str : string
        Plot suptitle
    scan_field_str : string
        Title of left subplot (original scan)
    radar_array_grid : array
        Interpolated reflectivity data for the requested scan
    fill_value : float
        Value used for unavailable data in the reflectivity scan
    results_mask : array
        Masks out reflectivity not contained in a roost ring
    center_coords : array
        Coordinates of centers of potential roost rings, in km from radar station
    figure_length : float, optional
        Figure size in inches   
        Default: figure_length = 6   
    filename_suffix : string, optional
        String to add to the end of plot filenames, if applicable
        Default: filename_suffix = ''
    """
    
    array_dim = np.shape(radar_array_grid)[0]
    cutoff_distance = (array_dim - 1) // 2
    
    ### plot the original scan
    
    fig, ax = plt.subplots(1, 2, figsize = (2.75 * figure_length, figure_length)) 
    
    original_mask = (radar_array_grid == fill_value)
    original_plot = ax[0].imshow(ma.array(radar_array_grid, mask = original_mask), interpolation = 'none', cmap = 'viridis', vmin = -20, vmax = 60)
    cbar = plt.colorbar(original_plot, ax = ax[0])
    cbar.set_label(label = 'Equivalent Reflectivity Factor (dBZ)', fontsize = 12)
    cbar.ax.tick_params(labelsize = 12)
    
    ax[0].set_xlabel('Distance (km)', fontsize = 12)
    ax[0].set_ylabel('Distance (km)', fontsize = 12)
    ax[0].set_title(scan_field_str, fontsize = 12)
    
    original_ticks = [50*i for i in range(1, 2*cutoff_distance//50)]
    new_xticks = [tick - cutoff_distance for tick in original_ticks]
    new_yticks = [cutoff_distance - tick for tick in original_ticks]
    ax[0].set_xticks(original_ticks, new_xticks)
    ax[0].set_yticks(original_ticks, new_yticks)
    ax[0].tick_params(labelsize = 12)
    
    ### plot the results
    
    ax[1].imshow(ma.array(radar_array_grid, mask = results_mask), interpolation = 'none', cmap = 'viridis', vmin = -20, vmax = 60)
    
    if len(center_coords) > 0:
        ax[1].scatter(center_coords[:, 0] + cutoff_distance, cutoff_distance - center_coords[:, 1], c = 'gold', edgecolor = 'k', s = 25, linewidth = 1.75) 
    
    ax[1].set_xlabel('Distance (km)', fontsize = 12)
    ax[1].set_ylabel('Distance (km)', fontsize = 12)
    
    results_title = 'Potential Roost Rings'
    ax[1].set_title(results_title, fontsize = 12)
    
    ax[1].set_xticks(original_ticks, new_xticks)
    ax[1].set_yticks(original_ticks, new_yticks)
    ax[1].tick_params(labelsize = 12)
    
    plt.subplots_adjust(wspace = 0.05)
    plt.suptitle(title_str, size = 16, y = 1.05)
    
    plt.savefig('results' + filename_suffix, bbox_inches = 'tight')
    plt.show()
    plt.close()

    
####################
    
    
# station location information
 
    
def get_station_latlon(station_str):
    
    """
    Get station latitude/longitude
    
    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    
    Returns
    -------
    station_lat : float
        Latitude
    station_lon : float
        Longitude
    """
    
    station_info = nexrad_common.get_nexrad_location(station_str)
    station_lat = station_info[0]
    station_lon = station_info[1]
    
    return station_lat, station_lon
    
    
def get_localtime_str(station_str, scan_dt):
    
    """
    Get formatted date and time in local time zone
    
    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    scan_dt : datetime
        Date and time of scan
    
    Returns
    -------
    localtime_str : string
        Formatted date and time in local time zone
    """
    
    station_lat, station_lon = get_station_latlon(station_str)
    station_tz = TimezoneFinder().timezone_at(lat = station_lat, lng = station_lon)
    dt_local = scan_dt.astimezone(timezone(station_tz))  
    localtime_str = dt_local.strftime('%B %d, %Y %I:%M %p %Z').replace(' 0', ' ')

    return localtime_str


####################

    
# dealing with missing data


def empty_roost_df():
    
    """
    No inputs.
    Returns empty DataFrame with same column names as output when roost rings are found.
    """
    
    return pd.DataFrame(columns = ['latitude', 'longitude', 'first detection', 'station name', 'year', 'month', 'day'])


def no_files_that_day(station_str, scan_date):

    """
    Preliminary check to see if any scan files are available for a given station and date

    Parameters
    ----------
    station_str : string
        Four letter identifier for the radar station
    scan_date : [year (int), month (int), day (int)]
        Date of the requested scan

    Returns
    -------
    no_files_tf : bool
        True if no files were found, False is some were
    """
    
    year_str = str(scan_date[0])

    month_str = str(scan_date[1])
    if len(month_str) == 1:
        month_str = '0' + month_str

    day_str = str(scan_date[2])
    if len(day_str) == 1:
        day_str = '0' + day_str

    path_str = year_str + "/" + month_str + "/" + day_str + "/" + station_str + "/"

    my_configuration = botocore.client.Config(signature_version = botocore.UNSIGNED)
    nexrad2_bucket = boto3.resource("s3", config = my_configuration).Bucket("noaa-nexrad-level2")
    n_files = len([available_file.key for available_file in nexrad2_bucket.objects.filter(Prefix = path_str)])
    no_files_tf = (n_files == 0)
    
    return no_files_tf