# RoostRingSearch
Python 3 package for finding swallow roost rings on US weather surveillance radar

**Melinda Kleczynski, Katie Bird, Chad Giusti, Jeffrey Buler**

## Language

[Python](https://www.python.org/)

## Dependencies

[Astral](https://astral.readthedocs.io/en/latest/)

[Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

[botocore](https://pypi.org/project/botocore/)

[datetime](https://docs.python.org/3/library/datetime.html)

[haversine](https://pypi.org/project/haversine/)

[Matplotlib](https://matplotlib.org/)

[NumPy](https://numpy.org/)

[pandas](https://pandas.pydata.org/)

[Py-ART](https://arm-doe.github.io/pyart/)

[pytz](https://pypi.org/project/pytz/)

[scikit-image](https://scikit-image.org/)

[SciPy](https://scipy.org/)

[tempfile](https://docs.python.org/3/library/tempfile.html)

[timezonefinder](https://pypi.org/project/timezonefinder/)

[warnings](https://docs.python.org/3/library/warnings.html)

## Dataset

[NEXRAD on AWS](https://registry.opendata.aws/noaa-nexrad)

## Selected resources

There are many excellent resources describing the use of weather radar data for monitoring birds and other animals in the airspace. I won't list all of them here, but I'll point out three in particular which were important in the development of this package.

The paper [Unlocking the Potential of NEXRAD Data through NOAA’s Big Data Partnership](https://journals.ametsoc.org/view/journals/bams/99/1/bams-d-16-0021.1.xml) discusses efforts to make the necessary data openly available.

The paper [The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.119/) describes the Py-ART package, which is instrumental to this work.

The paper [bioRad: biological analysis and visualization of weather radar data](https://onlinelibrary.wiley.com/doi/10.1111/ecog.04028) discusses extracting biological information from this type of data. 

## Package overview

### Getting started

Click on "Code" (near the top of the GitHub page, in green) and select "Download ZIP." Get the RoostRingSearch.py file. For example, if you're running a Jupyter notebook, move RoostRingSearch.py to the same folder as your notebook.

Make sure any packages listed under "Dependencies" are installed.

Run

```
import RoostRingSearch as rrs
```

and you should be all set! You may see some deprecation warnings, but that's okay.

If you're following along with the code in this readme, you'll also need to run

```
import numpy as np
import datetime
```

### Basic use

The two main functions are `morning_exodus` (checks several scans) and `find_roost_rings` (checks a single scan). Let's find some roost rings! Run the following (which will take a minute or two):

```
morning_df = rrs.morning_exodus('KDOX', [2020, 8, 23])
morning_df
```

If everything is set up correctly, you should see something like this:

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/readme_df.png" width = "700">

RoostRingSearch checked all the available scans from the KDOX radar during a window of time on the morning of August 23, 2020. It found four distinct roost rings and reported the latitude and longitude of the center of each ring, as well as the scan on which each ring was first detected.

Running the `morning_exodus` function like this is great if you just want the final results. If you want to check multiple days and/or stations, you can concatenate the DataFrames and save the results. If you're interested, RoostRingSearch can also display and/or return more information for further investigation. The functions `find_roost_rings` and `morning_exodus` both have the option to select `display_output = True` in order to see more information.

`morning_exodus` basically provides a framework for collecting the appropriate data, running `find_roost_rings` for each scan, and aggregating the results. Let's use the scan prefix information in `morning_df` to run `find_roost_rings` for each of the scans that `morning_exodus` flagged. 

```
for scan_prefix in np.unique(morning_df['scan prefix']):
    
    rrs.find_roost_rings(scan_prefix, display_output = True)
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_intro0.png" width = "700">

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_intro1.png" width = "700">

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_intro2.png" width = "700">

RoostRingSearch isn't perfect - it missed some roost rings on the middle scan! But it did a better job on the next scan. RoostRingSearch will make some mistakes, but it is a helpful tool if you would like an overview of roost ring locations without having to check all the scans manually.

### How RoostRingSearch works

#### Array processing

Consider the scan whose prefix is `2021/08/10/KDOX/KDOX20210810_1015`.

To check this scan for roost rings, run 

```
rrs.find_roost_rings('2021/08/10/KDOX/KDOX20210810_1015', display_output = True);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_array_processing.png" width = "700">

RoostRingSearch looks for roost rings in the reflectivity data. There is a lot of material on the reflectivity scan besides just roost rings. Some preprocessing helps clean up the array. RoostRingSearch performs the following preprocessing steps:

* Screen out low reflectivity
* If the clutter_filter_power_removed field is available, use it to screen out clutter
* If the cross_correlation_ratio field is available, use it to screen out precipitation

Here's what the reflectivity, clutter_filter_power_removed, and cross_correlation_ratio look like for this scan:

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/readme_3fields.jpg" width = "700">

RoostRingSearch can look for roost rings using only reflectivity data, but it doesn't perform as well. For example, precipitation can form shapes that are close enough to rings to be flagged as possible roost rings.

#### Linear filters

Once the reflectivity array is ready, RoostRingSearch uses linear filters to check for roost rings. Here's an example of what one of the filters looks like:

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/readme_filter.jpg" width = "300">

Here's the same filter shown over the reflectivity array for scale:

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/readme_filter_comparison.jpg" width = "300">

The search for roost rings involves comparing the filter to different parts of the processed reflectivity array. If there is enough reflectivity in the blue region of the filter and not too much reflectivity in the orange regions of the filter, there might be a roost ring there. There is some space between the blue and orange regions of the filter to help allow for roost rings which are irregularly shaped. This process is repeated for several filters since roost rings have varying widths and sizes.

### Parameters

`find_roost_rings` has some optional inputs. Four of these are parameters which control how the function decides whether or not it found a roost ring. These parameters and their default values are

```
cutoff_distance = 150
min_reflectivity = 0
max_background_noise = 0.05
min_signal = 0.3
```

We'll discuss how each of these work.

#### cutoff_distance

This parameter controls the region of the original scan which `find_roost_rings` checks for roost rings. The arrays are centered around the radar, and extend `cutoff_distance` km the the left (west), right (east), up (north), and down (south). A roost ring generally can't be identified unless the full circle which would be formed by the ring lies completely within the array. For example, with the default value `cutoff_distance = 150` the roost ring in the lower left corner of the following scan does not appear as a potential roost ring. (As a side note, the following line of code demonstrates an alternate input format for `find_roost_rings`.)

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2019, 8, 6, 10, 35)), display_output = True);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_cutoff_default.png" width = "700">

To find that roost ring, increase the default value of `cutoff_distance` to 200:

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2019, 8, 6, 10, 35)), display_output = True, 
                      cutoff_distance = 200);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_cutoff_200.png" width = "700">

Increasing the `cutoff_distance` creates larger arrays, which can increase the run time. Also, the data quality starts to decrease for large values of `cutoff_distance`. This scan demonstrates that if you increase the `cutoff_distance` to 250, the cross_correlation_ratio isn't available for the whole array. RoostRingSearch uses the cross_correlation_ratio to identify precipitation, so there can be more false positives due to precipitation in the corners of scans if you choose a `cutoff_distance` of 250 km.

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/readme_cutoff_precip.jpg" width = "700">

A scan such as this one can have a lot of precipitation, and still have a roost ring.

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2021, 8, 4, 9, 41)), display_output = True);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_cutoff_precip.png" width = "700">

#### min_reflectivity

Recall that one of the preprocessing steps is to screen out low reflectivity. The `min_reflectivity` parameter determines how low reflectivity needs to be in order to be removed from the array. Let's look at an example, starting with the default value of `min_reflectivity = 0`. Reflectivity values can be negative, so that's a nontrivial requirement. `find_roost_rings` only finds one of the roost rings.

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2021, 10, 1, 11, 20)), display_output = True);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_minrefl_default.png" width = "700">

Increasing the `min_reflectivity` to 10 finds two of the other rings instead, because they have higher reflectivity in the ring region and the noisy reflectivity in the background is screened out.

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2021, 10, 1, 11, 20)), display_output = True, 
                      min_reflectivity = 10);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_minrefl_10.png" width = "700">

Running the function multiple times with different values of `min_reflectivity` would find more roost rings, but at the cost of increased processing time. This could also increase the number of false positives (identifying something as a potential roost ring when it's really not a roost ring).

#### max_background_noise

Remember the linear filter? A roost ring should not have too much reflectivity in the orange regions. The `max_background_noise` parameter determines how much reflectivity is allowed. In this scan, the roost ring doesn't register with the default `max_background_noise = 0.05` because there is too much reflectivity around it.

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2021, 8, 7, 9, 40)), display_output = True);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_background_default.png" width = "700">

`find_roost_rings` is able to find this roost ring when `max_background_noise = 0.15`. Note that increasing the `max_background_noise` makes it easier for a region of a scan to count as a roost ring, so in general there will be more false positives.

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2021, 8, 7, 9, 40)), display_output = True, 
                      max_background_noise = 0.15);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_background_015.png" width = "700">

#### min_signal

Finally, the `min_signal` parameter determines how much reflectivity is needed in the blue region of the filter. Here's an example, starting with the default value `min_signal = 0.3`. There is a roost ring, but it doesn't have enough positive reflectivity for `find_roost_rings` to find it.

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2021, 7, 25, 9, 59)), display_output = True);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_minsignal_default.png" width = "700">

Decreasing the `min_signal` parameter to 0.25 relaxes the requirements enough to find this roost ring. As before, this makes false positive results more likely.

```
rrs.find_roost_rings(('KDOX', datetime.datetime(2021, 7, 25, 9, 59)), display_output = True, 
                      min_signal = 0.25);
```

<img src = "https://github.com/makleczy/RoostRingSearch/blob/main/readme_figures/results_readme_minsignal_025.png" width = "700">

### More info

Run `?rrs` to see some basic information about RoostRingSearch, including a list of all the functions that are available. You can also get information about any function. For example, running `?rrs.find_roost_rings` summarizes the `find_roost_rings` function.

## Documenting use

Kleczynski, M., Bird, K., Giusti, C., & Buler, J. (2022). RoostRingSearch (Version 1.0) [Computer software]. https://github.com/makleczy/RoostRingSearch

@software{RoostRingSearch,
<br>
author = {Kleczynski, Melinda and Bird, Katie and Giusti, Chad and Buler, Jeffrey},
<br>
month = {12},
<br>
title = {{RoostRingSearch}},
<br>
url = {https://github.com/makleczy/RoostRingSearch},
<br>
version = {v1.0},
<br>
year = {2022}
<br>
}
