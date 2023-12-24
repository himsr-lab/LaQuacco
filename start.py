import fnmatch
import math
import multiprocessing
import os
import platform
import random
import tifffile
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
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


def get_chan_stats(image, lmbdas=None):
    """Calculate the mean values and standard deviations for each of the image channels.

    Keyword arguments:
    images -- image file list
    """
    chan_stats = dict()
    name = os.path.basename(image)
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
            if lmbdas and chan < len(lmbdas):
                signal_min, _ = get_signal_min(pixels, lmbda[chan])
            else:
                signal_min, lmbda = get_signal_min(pixels)
            chan_stats[chan_name]["lmbda"] = lmbda
            # get signal mean
            chan_stats[chan_name]["signal_mean"] = np.mean(pixels[pixels >= signal_min])
            # get signal standard deviation
            chan_stats[chan_name]["signal_std"] = np.std(pixels[pixels >= signal_min])
            # get background mean
            chan_stats[chan_name]["backgr_mean"] = np.mean(pixels[pixels < signal_min])
            # get background standard deviation
            chan_stats[chan_name]["backgr_std"] = np.std(pixels[pixels < signal_min])
    return (image, chan_stats)


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
    q1 = np.percentile(array_norm, 25)  # Q1
    iqr = sp.stats.iqr(array_norm)  # IQR
    limit = q1 - 1.5 * iqr
    sig_min = sp.special.inv_boxcox(limit, lmbda)
    return (
        sig_min if not np.isnan(sig_min) else np.min(array[array > 0]),
        lmbda,
    )


PROCESSES = multiprocessing.cpu_count() // 2 or 1  # concurrent workers
# PROCESSES = 2

# main program
if __name__ == "__main__":
    # safe import of main module avoids spawning multiple processes simultaneously
    if platform.system() == "Windows":
        multiprocessing.freeze_support()  # required by 'multiprocessing'

    files = get_files(
        path=r"/Users/christianrickert/Desktop/MIBI", pat="*.tif", anti=""
    )
    sampling_perc = 1
    sampling_size = math.ceil(sampling_perc / 100 * len(files)) or 1
    samples = random.sample(files, sampling_size)
    channel_stats: Dict[str, Any] = dict()
    sample_args = [(sample, None) for sample in samples]
    with multiprocessing.Pool(PROCESSES) as pool:
        results = pool.starmap(get_chan_stats, sample_args)
    channel_stats = {sample: result for sample, result in results}
    # channel_stats = dict(sorted(channel_stats.items()))
    for image, channel_stat in channel_stats.items():
        print(f"{image}\n{channel_stat}", end="\n")
