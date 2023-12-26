import datetime
import fnmatch
import math
import multiprocessing
import os
import platform
import random
import tifffile
import xmltodict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


# functions
def calculate_boxplot(array):
    """Calculate the data for a Matplolib boxplot and
    immediately close the corresponding plot window:
    Accessing the data of the boxplot within a
    processing pool can crash Python.

    Keyword arguments:
    array  -- a Numpy array to be analyzed
    """
    boxplt = plt.boxplot(array)
    plt.close()
    return boxplt


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


def get_channel(page):
    """Get the channel name from a TIFF page.

    Keyword arguments:
    page -- the TIFF page
    """
    try:
        channel = page.tags["PageName"].value  # regular TIFF
    except KeyError:
        img_descr = page.tags["ImageDescription"].value  # OME-TIFF
        img_dict = xmltodict.parse(img_descr)
        vendor_id = next(iter(img_dict))  # only key
        try:
            channel = img_dict[vendor_id]["Name"]
        except KeyError:
            channel = None
    return channel


def get_chans_stats_means(chans_stats, chan, stats):
    """Returns the mean values from a list of channel data dictionaries.

    Keyword arguments:
    chans_stats -- a list of channel statistics dictionaries
    chan -- the key determining the channel value
    stats -- the key determining the channel's statistics
    """
    means = []
    for img_chans_data in chans_stats.values():
        if isinstance(img_chans_data, dict) and chan in img_chans_data:
            means.append(img_chans_data[chan][stats])
        else:  # channel missing in image
            means.append(None)
    # convert to Numpy array, keep Python datatype
    means = np.array(means, dtype="float")
    means[means == None] = np.nan
    return means


def get_colormap(count):
    """Return a colormap with `counts` number of colors.

    Keyword arguments:
    count  -- number of colors to be generated
    """
    color_points = np.linspace(0, 1, count)
    return [cm.hsv(color_point) for color_point in color_points]


def get_img_data(image, norm_lmbdas=None):
    """Calculate the mean values and standard deviations for each of the image channels.

    Keyword arguments:
    images -- image file list
    """
    img_chans_data = dict()
    img_name = os.path.basename(image)
    print(f"\tSAMPLE: {img_name}", flush=True)
    # open TIFF file to extract image information
    with tifffile.TiffFile(image) as tif:
        date_time = None
        series = tif.series
        pages = series[0].shape[0]
        # access all pages of the first series
        for p, page in enumerate(tif.pages[0:pages]):
            # identify channel by name
            chan = get_channel(page)
            if not chan:
                chan = str(p)
            # get date and time of acquisition
            if not date_time:
                date_time = get_timestamp(page.tags["DateTime"].value)
                date_time = img_chans_data["metadata"] = {"date_time": date_time}
            # prepare channel statistics
            if chan not in img_chans_data:
                img_chans_data[chan] = {}
            # get pixel data as flattend Numpy array
            pixls = page.asarray().flatten()
            # transform pixel data to be normally distributed
            if norm_lmbdas and len(norm_lmbdas) > p:
                lmbda = norm_lmbdas[p]
                norms, _ = power_transform(pixls, lmbda=lmbda)
            else:  # lambda not yet determined, computationally expensive
                norms, lmbda = power_transform(pixls)
                img_chans_data[chan]["sign_lmbda"] = lmbda
            # identify background as outliers from normally distributed signal
            boxplt_data = calculate_boxplot(norms)
            norms_sign_min = boxplt_data["whiskers"][0].get_ydata()[1]
            pixls_sign_min = sp.special.inv_boxcox(norms_sign_min, lmbda)
            img_chans_data[chan]["sign_min"] = pixls_sign_min
            # get basic statistics for signal
            sign_mean, sign_stdev, sign_stderr, sign_boxplt = get_stats(
                pixls[pixls >= pixls_sign_min]
            )
            img_chans_data[chan]["sign_mean"] = sign_mean
            img_chans_data[chan]["sign_stdev"] = sign_stdev
            img_chans_data[chan]["sign_stderr"] = sign_stderr
            img_chans_data[chan]["sign_bxplt"] = sign_boxplt
            # get basic statistics for background
            bckg_mean, bckg_stdev, bckg_stderr, bckg_boxplt = get_stats(
                pixls[pixls < pixls_sign_min]
            )
            img_chans_data[chan]["bckg_mean"] = bckg_mean
            img_chans_data[chan]["bckg_stdev"] = bckg_stdev
            img_chans_data[chan]["bckg_stderr"] = bckg_stderr
            img_chans_data[chan]["bckg_bxplt"] = bckg_boxplt
        return (image, img_chans_data)


def get_samples(population=None, perc=100):
    """From a list of elements, get a fractional subset of the data.

    Keyword arguments:
    population -- the list to take the samples from
    perc -- the percentage the subset represents
    """
    size = math.ceil(perc / 100 * len(population)) or 1
    samples = random.sample(population, size)
    return samples


def get_signal_min(array, lmbda=None):
    """Return the minimum signal value above background. First, we find the best transformation of
    all positive array values so that the data will be normally distributed. Second, we calculate
    the (bottom) box plots statistics for the normally distributed data.
    In effect, we're defining the bottom outliers of the data distribution as background.

    Keyword arguments:
    array  -- a Numpy array to be normalized
    """
    array_norm, lmbda = power_transform(array, lmbda)
    lower_fourth = np.percentile(array_norm, 25)  # Q1
    interquartile_range = sp.stats.iqr(array_norm)  # IQR
    low_extr = lower_fourth - 1.5 * interquartile_range
    sig_min = sp.special.inv_boxcox(low_extr, lmbda)
    np.min(array_norm[array_norm >= sig_min])
    return (
        sig_min if not np.isnan(sig_min) else np.min(array[array > 0]),
        lmbda,
    )


def get_stats(array):
    """Calculates basic statistics for the array.

    Keyword arguments:
    array -- Numpy array
    """
    mean = np.mean(array)
    stdev = np.std(array, ddof=1)  # estimating arithmetic mean
    stderr = get_stderr(array)
    boxplt = calculate_boxplot(array)
    return (mean, stdev, stderr, boxplt)


def get_stderr(array):
    """Calculates the standard error of the mean. We're estimating the
    arithmetic mean, so we're losing one degree of freedom (n - k).

    Keyword arguments:
    array -- Numpy array
    """
    if array.size:
        return np.sqrt(np.var(array, ddof=1) / array.size)
    else:
        return np.nan


def get_timestamp(timestamp):
    """Get a timestamp from a corresponding TIFF tag string.

    Keyword arguments:
    timestamp  -- the timestamp string
    """
    return datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")


def power_transform(array, lmbda=None):
    """Power-transforms a Numpy array with the `numpy.stats.boxcox` function.
    If the value of `lambda` is not given (None), the fuction will run an
    optimization routine to find the optimal value.
    The data needs to be positive (above zero) before the transformation.

    Keyword arguments:
    array  -- the untransformed Numpy array
    lambda  -- the transformation parameter
    """
    array_trans, lmbda, *_ = sp.stats.boxcox(array[array > 0], lmbda=lmbda)
    return (array_trans, lmbda)


processes = multiprocessing.cpu_count() // 2 or 1  # concurrent workers

# main program
if __name__ == "__main__":
    # safe import of main module avoids spawning multiple processes simultaneously
    if platform.system() == "Windows":
        multiprocessing.freeze_support()  # required by 'multiprocessing'
    # get a list of all image files
    files = get_files(
        path=r"/Users/christianrickert/Desktop/Polaris",
        # path=r"/Users/christianrickert/Desktop/MIBI",
        pat="*.tif",
        anti="",
    )
    # get a sample of the image files
    samples = get_samples(population=files, perc=0)
    # analyze the sample
    sample_args = [(sample, None) for sample in samples]
    with multiprocessing.Pool(processes) as pool:
        pool_results = pool.starmap(get_img_data, sample_args)
    # print(pool_results)
    pool_results = {sample: result for sample, result in pool_results}
    # for sample, result in pool_results.items():
    #    print(f"\n{sample}\n{result}")
    # pool_results = dict(sorted(pool_results.items()))
    # for image, channel_stat in pool_results.items():
    #    print(f"{image}\n{channel_stat}", end="\n")
    channels = [list(channel_stats.keys()) for channel_stats in pool_results.values()][
        0
    ]

    # print(pool_results[next(iter(pool_results))][chans[0]])

    # prepare colormap
    color_map = get_colormap(len(channels))

    # create figure and axes
    fig, ax = plt.subplots()
    # last_values = []
    means = []
"""
    for index, channel in enumerate(channels):
        # get statistics summary
        signal_means = get_chans_stats_means(pool_results, channel, "signal_mean")
        signal_mean = np.nanmean(signal_means)
        signal_stds = get_chans_stats_means(pool_results, channel, "signal_std")
        signal_errs = get_chans_stats_means(pool_results, channel, "signal_err")
        signal_std = np.nanmean(signal_stds)
        background_means = get_chans_stats_means(
            pool_results, channel, "background_mean"
        )
        background_mean = np.nanmean(background_means)
        background_stds = get_chans_stats_means(pool_results, channel, "background_std")
        background_errs = get_chans_stats_means(pool_results, channel, "background_err")
        background_std = np.nanmean(background_stds)

        # def draw_levey_jennings_plot():
        #    pass

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
        #
        # last_values.append(signal_means[-1])
        # ax.errorbar(
        #    range(len(signal_means)),
        #    signal_means,
        #    yerr=signal_errs,
        #    fmt="o-",
        #    color=color_map[index],
        #    label=channel + " [SIG]",
        # )
        means.append(signal_means)

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
        # last_values.append(background_means[-1])
        # ax.errorbar(
        #    range(len(background_means)),
        #    background_means,
        #    yerr=background_errs,
        #    fmt=".--",
        #    color=color_map[index],
        #    label=channel + " [BGR]",
        # )

    # order legend elements by first value plotted
    # handles, labels = plt.gca().get_legend_handles_labels()
    # zipped_legends = zip(handles, labels, last_values)
    # sorted_legends = sorted(zipped_legends, key=lambda l: l[-1], reverse=True)
    # handles, labels, _ = zip(*sorted_legends)
    # draw legend
    # legend = ax.legend(
    #    handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small"
    # )
    # adjust drawing area to fit legend
    # legend_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # legend_bbox_width = legend_bbox.width / fig.get_size_inches()[0]
    # fig.subplots_adjust(
    #    left=0.075, right=(0.95 - legend_bbox_width), wspace=0.05, hspace=0.05
    # )
    # plt.subplots_adjust(left=0.2)

    # sorted_items = sorted(files_dict.items(), key=lambda x: x[1])
    # sorted_filenames = [item[0] for item in sorted_items]

    print(pool_results.keys())
    sorted_samples = dict(sorted(pool_results.items(), key=lambda v: v[1]["date_time"]))
    for sorted_sample in sorted_samples:
        print(f"{sorted_sample}\n")
    #    for sample in sorted_samples:
    #        print(f"{sample} -> {pool_results['dsDNA (89)']['date_time']}")
    bp = ax.violinplot(means, showmeans=False, showmedians=False, showextrema=False)
    bp = ax.boxplot(means, meanline=True, showmeans=True)
    ax.set_xticks(
        [x for x in range(1, len(channels) + 1)],
        labels=channels,
        rotation=90,
        fontsize="small",
    )
    # plt.yscale("log")
    #    yticks = [0, 0.1]
    # ymax = np.nanmax(means)
    #    while max(yticks) < np.max(means):
    #        yticks.append(yticks[-1] * 10.0)
    #    plt.gca().set_yticks(yticks)  # Set the ticks positions
    #    plt.gca().set_yticklabels([str(ytick) for ytick in yticks])
    plt.show()
"""
