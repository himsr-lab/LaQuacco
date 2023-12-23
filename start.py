import fnmatch
import math
import os
import tifffile
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sp
from typing import Any, Dict


# functions
def get_files(path="", pat=None, anti=None, recurse=True):
    """Iterate through all files in a folder structure and
    return a list of matching files.

    Keyword arguments:
    path -- the path to a folder containing files (default "")
    pat -- string pattern that needs to be part of the file name (default "None")
    anti -- string pattern that may not be part of the file name (default "None")
    recurse -- boolen that allows the function to work recursively (default "False")
    """
    FILES = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file)
            if fnmatch.fnmatch(file, pat) and not fnmatch.fnmatch(file, anti):
                FILES.append(file)
        if not recurse:
            break  # from `os.walk()`
    return FILES


def get_signal_threshold(array):
    """Return the threshold value for separating background values from signa valuesl.
    In short, we're transforming the positive array values to be normally distributed.
    From this distribution picture a boxplot, where we're treating the background values
    as low-value "outliers" from our high-value signals.
    Keyword arguments:
    array  -- a Numpy array to be normalized
    """
    array_norm, lmbda = sp.stats.boxcox(array[array > 0])  # positive values only
    quartile_one = np.percentile(array_norm, 25)  # Q1
    interquartile_range = sp.stats.iqr(array_norm)  # IQR
    bottom_whisker = quartile_one - 1.5 * interquartile_range
    signal_threshold = sp.special.inv_boxcox(bottom_whisker, lmbda)
    return (
        signal_threshold if not np.isnan(signal_threshold) else np.min(array[array > 0])
    )


def extend_dict_list(dictionary, key, value):
    """Appends a value to a dictionary's list, if the key already exists.
    Creates the key and sets the value, if the key has been missing.
    Keyword arguments:
    dictionary  -- an existing dictionary
    key  -- key for the list to be created
    value -- value to add to the list
    """
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)


files = get_files(path=r"/Users/christianrickert/Desktop/Polaris", pat="*.tif", anti="")
sampling_percentage = 25
sampling_size = math.ceil(sampling_percentage / 100 * len(files)) or 1
samples = random.sample(files, sampling_size)
print(f"{len(files)}, {len(samples)}")

channel_stats: Dict[str, Any] = dict()

for file in sorted(samples):
    name = os.path.basename(file)
    print(f"\n\tFILE: {name}", flush=True)

    # open TIFF file to extract image information
    with tifffile.TiffFile(file) as tif:
        # use data from first series (of pages) only
        series = tif.series
        pages = series[0].shape[0]
        # access pages of first series
        for chan, page in enumerate(tif.pages[0:pages]):
            # identify channel with page name
            channel_name = None
            try:
                channel_name = page.tags["PageName"].value  # regular TIFF
            except KeyError:
                image_description = page.tags["ImageDescription"].value  # OME-TIFF
                image_dictionary = xmltodict.parse(image_description)
                vendor_id = next(iter(image_dictionary))  # first and only key
                try:
                    channel_name = image_dictionary[vendor_id]["Name"]
                except KeyError:
                    channel_name = str(chan)
            # get pixel data as flattend Numpy array
            pixels = page.asarray().flatten()
            # prepare channel statistics
            if channel_name not in channel_stats:
                channel_stats[channel_name] = {}
            # get minimum signal value (threshold)
            signal_min = get_signal_threshold(pixels)
            # get signal mean
            signal_mean = np.mean(pixels[pixels >= signal_min])
            extend_dict_list(channel_stats[channel_name], "signal_mean", signal_mean)
            # get signal standard deviation
            signal_std = np.std(pixels[pixels >= signal_min])
            extend_dict_list(channel_stats[channel_name], "signal_std", signal_std)
            # get background mean
            backgr_mean = np.mean(pixels[pixels < signal_min])
            extend_dict_list(channel_stats[channel_name], "backgr_mean", backgr_mean)
            # get background standard deviation
            backgr_std = np.std(pixels[pixels < signal_min])
            extend_dict_list(channel_stats[channel_name], "background_std", backgr_std)

            # print(f"{channel_name}:\t {get_signal_threshold(pixels)}")
            # pixels = np.log(pixels[pixels > 0])
            # plt.hist(pixels, bins=int(10 * np.max(pixels)), color="black")
            # plt.axvline(x=get_signal_threshold(pixels), color="green", linestyle="--")
            # plt.axvline(
            #    x=np.percentile(pixels[pixels > 0], 10), color="red", linestyle="--"
            # )
            # plt.show()
            # print(np.mean(sorted_pixels[: get_bottom_index(sorted_pixels, 10)]))
            # print(np.mean(sorted_pixels[get_top_index(sorted_pixels, 20) :]))
        print(channel_stats)
