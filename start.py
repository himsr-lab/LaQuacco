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
    means = []
    for chan_stats in chans_stats.values():
        if chan in chan_stats:
            means.append(chan_stats[chan][stats])
        else:  # channel missing in image
            means.append(None)
    # convert to Numpy array, keep Python datatype
    means = np.array(means, dtype="float")
    means[means == None] = np.nan
    return means


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
            chan_stats[chan_name]["background_mean"] = np.mean(
                pixels[pixels < signal_min]
            )
            # get background standard deviation
            chan_stats[chan_name]["background_std"] = np.std(
                pixels[pixels < signal_min], ddof=1
            )
            # get standard error of the mean
            chan_stats[chan_name]["background_err"] = np.sqrt(
                chan_stats[chan_name]["background_std"]
                / len(pixels[pixels >= signal_min])
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
        path=r"/Users/christianrickert/Desktop/MIBI",
        pat="*.tif",
        anti="",
        # path=r"/Users/christianrickert/Desktop/MIBI",
        # pat="*.tif",
        # anti="",
    )
    # get a sample of the image files
    sampling_perc = 0.5
    # sampling_perc = 1
    sampling_size = math.ceil(sampling_perc / 100 * len(files)) or 1
    samples = random.sample(files, sampling_size)
    # analyze the sample
    samples_results: Dict[str, Any] = dict()
    sample_args = [(sample, None) for sample in samples]
    with multiprocessing.Pool(processes) as pool:
        results = pool.starmap(get_chan_stats, sample_args)
    samples_results = {sample: result for sample, result in results}
    # for sample, result in samples_results.items():
    #    print(f"\n{sample}\n{result}")
    # samples_results = dict(sorted(samples_results.items()))
    # for image, channel_stat in samples_results.items():
    #    print(f"{image}\n{channel_stat}", end="\n")
    channels = [
        list(channel_stats.keys()) for channel_stats in samples_results.values()
    ][0]

    for channel in channels:
        print(repr(channel))
    # print(samples_results[next(iter(samples_results))][chans[0]])

    # channel = "DAPI (DAPI)"
    # channel = "dsDNA (89)"

    # prepare colormap
    color_map = get_colormap_values(len(channels))

    # create figure and axes
    fig, ax = plt.subplots()
    last_values = []

    for index, channel in enumerate(channels):
        # get statistics summary
        signal_means = get_chans_stats_means(samples_results, channel, "signal_mean")
        signal_mean = statistics.mean(signal_means)
        signal_stds = get_chans_stats_means(samples_results, channel, "signal_std")
        signal_errs = get_chans_stats_means(samples_results, channel, "signal_err")
        signal_std = statistics.mean(signal_stds)
        background_means = get_chans_stats_means(
            samples_results, channel, "background_mean"
        )
        background_mean = statistics.mean(background_means)
        background_stds = get_chans_stats_means(
            samples_results, channel, "background_std"
        )
        background_errs = get_chans_stats_means(
            samples_results, channel, "background_err"
        )
        background_std = statistics.mean(background_stds)

        def draw_levey_jennings_plot():
            pass

        #        if signal_mean - 2 * signal_std > 0:
        #            ax.axhline(
        #                y=signal_mean - 2 * signal_std,
        #                color=color_map[index],
        #                linestyle="dotted",
        #            )
        #        if signal_mean - signal_std > 0:
        #            ax.axhline(
        #                y=signal_mean - signal_std, color=color_map[index], linestyle="dashed"
        #            )
        #        ax.axhline(y=signal_mean, color=color_map[index], linestyle="dashdot")
        #        ax.axhline(
        #            y=signal_mean + signal_std, color=color_map[index], linestyle="dashed"
        #        )
        #        ax.axhline(
        #            y=signal_mean + 2 * signal_std, color=color_map[index], linestyle="dotted"
        #        )
        # ax.plot(signal_means, color=color_map[index], linestyle="solid")
        last_values.append(signal_means[-1])
        ax.errorbar(
            range(len(signal_means)),
            signal_means,
            yerr=signal_errs,
            fmt="o-",
            color=color_map[index],
            label=channel + " [SIG]",
        )

        # plot statistics summary
        #        if background_mean - 2 * background_std > 0:
        #            ax.axhline(
        #                y=background_mean - 2 * background_std,
        #                color=color_map[index],
        #                linestyle="dotted",
        #            )
        #        if background_mean - background_std > 0:
        #            ax.axhline(
        #                y=background_mean - background_std, color=color_map[index], linestyle="dashed"
        #            )
        #        ax.axhline(y=background_mean, color=color_map[index], linestyle="dashdot")
        #        ax.axhline(
        #            y=background_mean + background_std, color=color_map[index], linestyle="dashed"
        #        )
        #        ax.axhline(
        #            y=background_mean + 2 * background_std, color=color_map[index], linestyle="dotted"
        #        )
        # ax.plot(background_means, color=color_map[index], linestyle="solid")
        # plt.subplots_adjust(left=0.30)
        # ax.text(-1, signal_means[0], channel, color=color_map[index])
        last_values.append(background_means[-1])
        ax.errorbar(
            range(len(background_means)),
            background_means,
            yerr=background_errs,
            fmt=".--",
            color=color_map[index],
            label=channel + " [BGR]",
        )

    # order legend elements by first value plotted
    handles, labels = plt.gca().get_legend_handles_labels()
    zipped_legends = zip(handles, labels, last_values)
    sorted_legends = sorted(zipped_legends, key=lambda l: l[-1], reverse=True)
    handles, labels, _ = zip(*sorted_legends)
    # draw legend
    legend = ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small"
    )
    # adjust drawing area to fit legend
    legend_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    legend_bbox_width = legend_bbox.width / fig.get_size_inches()[0]
    fig.subplots_adjust(
        left=0.075, right=(0.95 - legend_bbox_width), wspace=0.05, hspace=0.05
    )
    # plt.subplots_adjust(left=0.2)
    plt.show()
