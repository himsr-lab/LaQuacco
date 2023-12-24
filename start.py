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


def get_chan_stats(images):
    """Calculate the mean values and standard deviations for each of the image channels.

    Keyword arguments:
    images -- image file list
    """
    chan_stats = dict()
    for image in images:
        name = os.path.basename(image)
        extend_dict_list(chan_stats, "images", name)
        print(f"\tSAMPLE: {name}", flush=True)
        # open TIFF file to extract image information
        with tifffile.TiffFile(image) as tif:
            # use data from first series (of pages) only
            series = tif.series
            pages = series[0].shape[0]
            # access pages of first series
            for chan, page in enumerate(tif.pages[0:pages]):
                # identify channel with page name
                chan_name = None
                try:
                    chan_name = page.tags["PageName"].value  # regular TIFF
                except KeyError:
                    img_descr = page.tags["ImageDescription"].value  # OME-TIFF
                    img_dict = xmltodict.parse(img_descr)
                    vendor_id = next(iter(img_dict))  # first and only key
                    try:
                        chan_name = img_dict[vendor_id]["Name"]
                    except KeyError:
                        chan_name = str(chan)
                # get pixel data as flattend Numpy array
                pixels = page.asarray().flatten()
                # prepare channel statistics
                if chan_name not in chan_stats:
                    chan_stats[chan_name] = {}
                # get minimum signal value (threshold) and boxcox lambda
                if "boxcox_lmbda" in chan_stats[chan_name]:
                    signal_min, _ = get_signal_min(
                        pixels, lmbda=chan_stats[chan_name]["boxcox_lmbda"]
                    )
                else:  # get lambda from first channels of first image
                    signal_min, lmbda = get_signal_min(pixels)
                    chan_stats[chan_name]["boxcox_lmbda"] = lmbda
                # get signal mean
                extend_dict_list(
                    chan_stats[chan_name],
                    "signal_mean",
                    np.mean(pixels[pixels >= signal_min]),
                )
                # get signal standard deviation
                extend_dict_list(
                    chan_stats[chan_name],
                    "signal_std",
                    np.std(pixels[pixels >= signal_min]),
                )
                # get background mean
                extend_dict_list(
                    chan_stats[chan_name],
                    "backgr_mean",
                    np.mean(pixels[pixels < signal_min]),
                )
                # get background standard deviation
                extend_dict_list(
                    chan_stats[chan_name],
                    "backgr_std",
                    np.std(pixels[pixels < signal_min]),
                )
    return chan_stats


def get_signal_min(array, lmbda=None):
    """Return the minimum signal value above background. First, we find the best transformation of
    all positive array values so that the data will be normally distributed. Second, we calculate
    the (bottom) box plots statistics for the normally distributed data.
    In effect, we're defining the bottom outliers of the data distribution as background.
    Keyword arguments:
    array  -- a Numpy array to be normalized
    """
    array_norm, lmbda, *_ = sp.stats.boxcox(
        array[array > 0], lmbda=lmbda
    )  # computationally expensive, when lambda is unknown
    quartile_one = np.percentile(array_norm, 25)  # Q1
    interquartile_range = sp.stats.iqr(array_norm)  # IQR
    bottom_whisker = quartile_one - 1.5 * interquartile_range
    signal_threshold = sp.special.inv_boxcox(bottom_whisker, lmbda)
    return (
        signal_threshold
        if not np.isnan(signal_threshold)
        else np.min(array[array > 0]),
        lmbda,
    )


files = get_files(path=r"/Users/christianrickert/Desktop/Polaris", pat="*.tif", anti="")
sampling_perc = 20
sampling_size = math.ceil(sampling_perc / 100 * len(files)) or 1
samples = random.sample(files, sampling_size)

channel_stats: Dict[str, Any] = dict()

channel_stats = get_chan_stats(sorted(samples))
# print(channel_stats)

for file in sorted(files):
    if file not in channel_stats["images"]:
        name = os.path.basename(file)
        print(f"\tIMAGE: {name}", flush=True)

    # print(f"{channel_name}:\t {get_signal_min(pixels)}")
    # pixels = np.log(pixels[pixels > 0])
    # plt.hist(pixels, bins=int(10 * np.max(pixels)), color="black")
    # plt.axvline(x=get_signal_min(pixels), color="green", linestyle="--")
    # plt.axvline(
    #    x=np.percentile(pixels[pixels > 0], 10), color="red", linestyle="--"
    # )
    # plt.show()
    # print(np.mean(sorted_pixels[: get_bottom_index(sorted_pixels, 10)]))
    # print(np.mean(sorted_pixels[get_top_index(sorted_pixels, 20) :]))
