import fnmatch
import math
import multiprocessing
import os
import platform
import random
import statistics
import tifffile
import xmltodict
import matplotlib.cm as cm
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


def get_chans_stats_means(chans_stats, chan, stats):
    """Returns the mean values from a list of dictionaries
    for a given channel and statistics.

    Keyword arguments:
    chans_stats -- a list of channel statistics dictionaries
    chan -- the key determining the channel value
    stats -- the key determining the channel's statistics
    """
    return [chan_stats[chan][stats] for chan_stats in chans_stats.values()]


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
            chan_stats[chan_name]["signal_std"] = np.std(
                pixels[pixels >= signal_min], ddof=1
            )
            # get standard error of the mean
            chan_stats[chan_name]["signal_err"] = np.sqrt(
                chan_stats[chan_name]["signal_std"] / len(pixels[pixels >= signal_min])
            )
            # get background mean
            chan_stats[chan_name]["backgr_mean"] = np.mean(pixels[pixels < signal_min])
            # get background standard deviation
            chan_stats[chan_name]["backgr_std"] = np.std(
                pixels[pixels < signal_min], ddof=1
            )
            # get standard error of the mean
            chan_stats[chan_name]["backgr_err"] = np.sqrt(
                chan_stats[chan_name]["backgr_std"] / len(pixels[pixels >= signal_min])
            )
    return (image, chan_stats)


def get_colormap_values(count):
    """Return a colormap with `counts` number of colors.
    Keyword arguments:
    count  -- number of colors to be generated
    """
    color_points = np.linspace(0, 1, count)
    return [cm.hsv(color_point) for color_point in color_points]


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


processes = multiprocessing.cpu_count()  # // 2 or 1  # concurrent workers

# main program
if __name__ == "__main__":
    # safe import of main module avoids spawning multiple processes simultaneously
    if platform.system() == "Windows":
        multiprocessing.freeze_support()  # required by 'multiprocessing'
    # get a list of all image files
    files = get_files(
        path=r"/Users/christianrickert/Desktop/Polaris",
        pat="*.tif",
        anti="",
        # path=r"/Users/christianrickert/Desktop/MIBI",
        # pat="*.tif",
        # anti="",
    )
    # get a sample of the image files
    sampling_perc = 30
    # sampling_perc = 1
    sampling_size = math.ceil(sampling_perc / 100 * len(files)) or 1
    samples = random.sample(files, sampling_size)
    # analyze the sample
    channels_stats: Dict[str, Any] = dict()
    sample_args = [(sample, None) for sample in samples]
    with multiprocessing.Pool(processes) as pool:
        results = pool.starmap(get_chan_stats, sample_args)
    channels_stats = {sample: result for sample, result in results}

    # channels_stats = dict(sorted(channels_stats.items()))
    # for image, channel_stat in channels_stats.items():
    #    print(f"{image}\n{channel_stat}", end="\n")
    chans = [list(channel_stats.keys()) for channel_stats in channels_stats.values()][0]
    # print(chans)
    # print(channels_stats[next(iter(channels_stats))][chans[0]])

    channel = "DAPI (DAPI)"
    # channel = "dsDNA (89)"

    signal_means = get_chans_stats_means(channels_stats, channel, "signal_mean")
    signal_mean = statistics.mean(signal_means)
    signal_stds = get_chans_stats_means(channels_stats, channel, "signal_std")
    signal_errs = get_chans_stats_means(channels_stats, channel, "signal_err")
    signal_std = statistics.mean(signal_stds)
    backgr_means = get_chans_stats_means(channels_stats, channel, "backgr_mean")
    backgr_mean = statistics.mean(backgr_means)
    backgr_stds = get_chans_stats_means(channels_stats, channel, "backgr_std")
    backgr_errs = get_chans_stats_means(channels_stats, channel, "backgr_err")
    backgr_std = statistics.mean(backgr_stds)

    fig, axs = plt.subplots()

    if backgr_mean - 2 * backgr_std > 0:
        axs.axhline(y=backgr_mean - 2 * backgr_std, color="black", linestyle="dotted")
    if backgr_mean - backgr_std > 0:
        axs.axhline(y=backgr_mean - backgr_std, color="black", linestyle="dashed")
    axs.axhline(y=backgr_mean, color="black", linestyle="dashdot")
    axs.axhline(y=backgr_mean + backgr_std, color="black", linestyle="dashed")
    axs.axhline(y=backgr_mean + 2 * backgr_std, color="black", linestyle="dotted")
    # axs.plot(backgr_means, color="black", linestyle="solid")
    axs.errorbar(
        range(len(backgr_means)),
        backgr_means,
        yerr=backgr_errs,
        fmt="-o",
        color="black",
    )

    colormap = get_colormap_values(len(chans))

    def draw_levey_jennings_plot():
        pass

    if signal_mean - 2 * signal_std > 0:
        axs.axhline(y=signal_mean - 2 * signal_std, color="blue", linestyle="dotted")
    if signal_mean - signal_std > 0:
        axs.axhline(y=signal_mean - signal_std, color="blue", linestyle="dashed")
    axs.axhline(y=signal_mean, color="blue", linestyle="dashdot")
    axs.axhline(y=signal_mean + signal_std, color="blue", linestyle="dashed")
    axs.axhline(y=signal_mean + 2 * signal_std, color="blue", linestyle="dotted")
    # axs.plot(signal_means, color="blue", linestyle="solid")
    axs.errorbar(
        range(len(signal_means)),
        signal_means,
        yerr=signal_errs,
        fmt="-o",
        color=colormap[0],
    )

    plt.show()
