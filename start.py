import datetime
import fnmatch
import math
import multiprocessing
import os
import platform
import random
import sys
import tifffile
import xmltodict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


# functions
def boxcox_transform(array, lmbda=None):
    """Power-transforms a Numpy array with the `numpy.stats.boxcox` function.
    If the value of `lambda` is not given (None), the fuction will determine
    the optimal value that maximizes the log-likelihood function.
    The data needs to be positive (above zero).

    Keyword arguments:
    array  -- the untransformed Numpy array
    lambda  -- the transformation parameter
    """
    array = array[array > 0]
    if lmbda:
        boxcox = sp.stats.boxcox(array, lmbda=lmbda, alpha=None)
        maxlog = None
    else:
        boxcox = np.array(0)
        maxlog = sp.stats.boxcox_normmax(array, brack=(-0.25, 0.75), method="mle")
    return (boxcox, maxlog)


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


def get_files(path="", pat=None, anti=None, recurse=False):
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


def get_chan(page):
    """Get the channel name from a TIFF page.

    Keyword arguments:
    page -- the TIFF page
    """
    try:
        chan = page.tags["PageName"].value  # regular TIFF
    except KeyError:
        img_descr = page.tags["ImageDescription"].value  # OME-TIFF
        img_dict = xmltodict.parse(img_descr)
        vendor_id = next(iter(img_dict))  # only key
        try:
            chan = img_dict[vendor_id]["Name"]
        except KeyError:
            chan = None
    return chan


def get_chan_data(imgs_chans_data, chan, data):
    """Returns channel data from image data dictionaries.
    Works across all images to retrieve the channel data.

    Keyword arguments:
    imgs_chans_data -- dictionaries with image data
    chan -- the key determining the channel value
    data -- the key determining the channel data
    """
    chan_data = []
    for _img, chans_data in imgs_chans_data.items():
        if chan in chans_data and chan not in ["metadata"]:
            chan_data.append(chans_data[chan][data])
        else:  # channel missing in image
            chan_data.append(None)
    # convert to Numpy array, keep Python datatype
    chan_data = np.array(chan_data, dtype="float")
    chan_data[chan_data == None] = np.nan
    return chan_data


def get_colormap(count):
    """Return a colormap with `counts` number of colors.

    Keyword arguments:
    count  -- number of colors to be generated
    """
    color_points = np.linspace(0, 1, count)
    return [cm.hsv(color_point) for color_point in color_points]

    for i, image in enumerate(images_img_data):
        for chan in chans:
            chans_means[i] += images_img_data[image][chan]["sign_mean"]
            chans_stderrs[i] += images_img_data[image][chan]["sign_stderr"]


def get_img_data(imgs_chans_data, img, data):
    """Returns image data from image data dictionaries.
    Works across all channels to retrieve the image data.

    Keyword arguments:
    imgs_chans_data -- dictionaries with image data
    img -- the key determining the image value
    data -- the key determining the channel data
    """
    img_data = []
    if img in imgs_chans_data:
        for chan in imgs_chans_data[img].values():
            if chan not in ["metadata"]:
                if data in chan:
                    img_data.append(chan[data])
                else:  # data missing in channel
                    img_data.append(None)
    else:
        img_data = None
    # convert to Numpy array, keep Python datatype
    img_data = np.array(img_data, dtype="float")
    img_data[img_data == None] = np.nan
    return img_data


def get_mean(array):
    """Returns the arithmetic mean.
    Numpy produces a `RuntimeError` when all elements of the array
    are `np.nan` values. Instead of returning an error, we return `np.nan`.


    Keyword arguments:
    array -- Numpy array
    """
    mean = np.nan
    if array.size > 0 and not np.all(np.isnan(array)):
        mean = np.nanmean(array)
    return mean


def get_run_slice(array, index, slice_margin, slice_min=True):
    """Returns the slice of an array centered at the index
     with a margin of elements included before and after.

    Keyword arguments:
    array -- Numpy array
    index  -- center position of the slice
    margin  -- element count before and after index
    """
    slice = np.array(0)
    if array.size > 0:
        slice = array[
            max(0, index - slice_margin) : min(index + slice_margin + 1, array.size)
        ]
    return slice


def get_samples(population=None, perc=100):
    """From a list of elements, get a fractional subset of the data.

    Keyword arguments:
    population -- the list to take the samples from
    perc -- the percentage the subset represents
    """
    size = math.ceil(perc / 100 * len(population)) or 1
    samples = random.sample(population, size)
    return samples


def get_stats(array):
    """Calculates basic statistics for the array.

    Keyword arguments:
    array -- Numpy array
    """
    mean = get_mean(array)
    stdev = get_stdev(array)
    stderr = get_stderr(array)
    return (mean, stdev, stderr)


def get_stderr(array, mean=None, ddof=1):
    """Calculates the standard error of the arithmetic mean.

    Keyword arguments:
    array -- Numpy array
    mean -- arithmetic mean of the samples
    est -- number of estimated parameters, reduces degrees of freedom
    """
    stderr = np.nan
    if array.size:
        stderr = np.sqrt(
            get_var(array, mean, ddof) / array.size
        )  # unreliability measure
    return stderr


def get_stdev(array, mean=None, ddof=1):
    """Calculates the standard deviation.

    Keyword arguments:
    array -- Numpy array
    mean -- arithmetic mean of the samples
    est -- number of estimated parameters, reduces degrees of freedom
    """
    stdev = np.nan
    if array.size:
        stdev = np.sqrt(get_var(array, mean, ddof))
    return stdev


def get_var(array, mean=None, ddof=1):
    """Calculates the variance - the variability of the data.
    If we have observed the whole population (all possible samples),
    then we would have as many degrees of freedom as observed samples.
    However, by default we only observe a fraction of all possilbe samples,
    so we lose one degree of freedom (n-k) - for estimating the arithmetic mean.

    Keyword arguments:
    array -- Numpy array
    mean -- arithmetic mean of the samples
    est -- number of estimated parameters, reduces degrees of freedom
    """
    variance = np.nan
    if array.size:
        if not mean:
            mean = get_mean(array)
        sums_squared = np.nansum(np.power(array - mean, 2))
        degrs_freedom = array.size - ddof
        if degrs_freedom > 0:
            variance = sums_squared / degrs_freedom
    return variance


def get_timestamp(timestamp):
    """Get a timestamp from a corresponding TIFF tag string.

    Keyword arguments:
    timestamp  -- the timestamp string
    """
    return datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")


def read_img_data(image, chan_lmbdas=None):
    """Calculate the mean values and standard deviations for each of the image channels.

    Keyword arguments:
    images -- image file list
    """
    img_chans_data = dict()
    img_name = os.path.basename(image)
    if chan_lmbdas:
        print(f"\tIMAGE: {img_name}", flush=True)
    else:
        print(f"\tSAMPLE: {img_name}", flush=True)
    # open TIFF file to extract image information
    with tifffile.TiffFile(image) as tif:
        date_time = None
        series = tif.series
        pages = series[0].shape[0]
        # access all pages of the first series
        for p, page in enumerate(tif.pages[0:pages]):
            # identify channel by name
            chan = get_chan(page)
            if not chan:
                chan = str(p)
            # prepare channel statistics
            if chan not in img_chans_data:
                img_chans_data[chan] = {}
            # get pixel data as Numpy array
            pixls = page.asarray()
            # power-transform data and get image statistics
            if chan_lmbdas:
                # get date and time of acquisition
                if not date_time:
                    date_time = get_timestamp(page.tags["DateTime"].value)
                    date_time = img_chans_data["metadata"] = {"date_time": date_time}
                # transform pixel data to be normally distributed
                norms, _ = boxcox_transform(pixls, lmbda=chan_lmbdas[chan])
                # identify background as outliers from normally distributed signal
                boxplt_data = calculate_boxplot(norms)
                norms_sign_min = boxplt_data["whiskers"][0].get_ydata()[1]
                pixls_sign_min = sp.special.inv_boxcox(
                    norms_sign_min, chan_lmbdas[chan]
                )
                if np.isnan(pixls_sign_min):  # bottom whisker missing
                    pixls_sign_min = 0.0
                img_chans_data[chan]["sign_min"] = pixls_sign_min
                # get basic statistics for signal
                (
                    img_chans_data[chan]["sign_mean"],
                    img_chans_data[chan]["sign_stdev"],
                    img_chans_data[chan]["sign_stderr"],
                ) = get_stats(pixls[pixls >= pixls_sign_min])
                # get basic statistics for background
                (
                    img_chans_data[chan]["bckg_mean"],
                    img_chans_data[chan]["bckg_stdev"],
                    img_chans_data[chan]["bckg_stderr"],
                ) = get_stats(pixls[pixls < pixls_sign_min])
            else:  # lambda not yet determined
                _, chan_lmbda = boxcox_transform(pixls)
                img_chans_data[chan]["chan_lmbda"] = chan_lmbda
        return (image, img_chans_data)


processes = multiprocessing.cpu_count() // 2 or 1  # concurrent workers


# main program
if __name__ == "__main__":
    # safe import of main module avoids spawning multiple processes simultaneously
    if platform.system() == "Windows":
        multiprocessing.freeze_support()  # required by 'multiprocessing'

    # get a list of all image files
    files = sorted(
        get_files(
            path=r"C:\Users\Christian Rickert\Desktop\Polaris",
            # path=r"/Users/christianrickert/Desktop/MIBI/UCD158/raw",
            pat="*.tif",
            anti="",
        ),
        key=str.lower,
    )

    # sample experimental image data
    try:
        samples = sorted(get_samples(population=files, perc=100), key=str.lower)
        sample_args = [(sample, None) for sample in samples]
    except ValueError:
        print("Could not draw samples from experimental population.")
        sys.exit(1)

    # analyze the sample data
    with multiprocessing.Pool(processes) as pool:
        sample_results = pool.starmap(read_img_data, sample_args)
        pool.close()  # wait for worker tasks to complete
        pool.join()  # wait for worker process to exit
    samples_img_data = {sample: img_data for (sample, img_data) in sample_results}

    chans_set = set()  # avoid duplicate entries
    for img_data in samples_img_data.values():
        for chan in img_data:
            if chan not in ["metadata"]:
                chans_set.add(chan)
    chans = sorted(chans_set, key=str.lower)

    # prepare colormap
    color_map = get_colormap(len(chans))

    # prepare lambdas for power transform
    chan_lmbdas = {}
    for chan in chans:
        chan_data = get_chan_data(samples_img_data, chan, "chan_lmbda")
        chan_mean = get_mean(chan_data)
        chan_lmbdas[chan] = chan_mean

    # analyze experimental image data
    image_args = [(image, chan_lmbdas) for image in files]
    with multiprocessing.Pool(processes) as pool:
        image_results = pool.starmap(read_img_data, image_args)
        pool.close()  # wait for worker tasks to complete
        pool.join()  # wait for worker process to exit
    images_img_data = {image: img_data for (image, img_data) in image_results}

    # sort experimental image data by time stamp
    images_img_data = dict(
        sorted(images_img_data.items(), key=lambda v: v[1]["metadata"]["date_time"])
    )

    # create figure and axes
    fig, ax = plt.subplots()

    """
    # distribution chart
    data_means = []
    data_norms = []
    for c, chan in enumerate(chans):
        # get statistics summary
        signal_means = get_chan_data(images_img_data, chan, "sign_mean")
        data_means.append(signal_means)
        data_norms.append(
            boxcox_transform(np.array(signal_means), lmbda=chan_lmbdas[chan])[0]
        )
    vp = ax.violinplot(
        data_means, showmeans=False, showmedians=False, showextrema=False
    )
    for v in vp["bodies"]:
        v.set_facecolor("black")
        v.set_edgecolor("black")
    bp = ax.boxplot(data_norms, meanline=True, showmeans=True)
    for b in bp["medians"]:
        b.set_color("black")
    for b in bp["means"]:
        b.set_color("black")
        b.set_linestyle("dashed")
    ax.set_xticks(
        [x for x in range(1, len(chans) + 1)],
        labels=chans,
        rotation=90,
        fontsize="small",
    )
    plt.show()
    """

    """
    # channels chart
    data_lasts = []
    signal_labels = [os.path.basename(image) for image in images_img_data.keys()]
    for c, chan in enumerate(chans):
        # get image statistics
        signal_means = get_chan_data(images_img_data, chan, "sign_mean")
        data_lasts.append(signal_means[-1])
        signal_stderrs = get_chan_data(images_img_data, chan, "sign_stderr")
        ax.errorbar(
            signal_labels,
            signal_means,
            yerr=signal_stderrs,
            fmt="o-",
            linewidth=1,
            markersize=2,
            color=color_map[c],
            label=chan + " [SIG]",
        )
    img_means = []
    img_stderrs = []
    for img in images_img_data:
        img_means.append(get_mean(get_img_data(images_img_data, img, "sign_mean")))
        img_stderrs.append(get_mean(get_img_data(images_img_data, img, "sign_stderr")))
    data_lasts.append(img_means[-1])
    ax.errorbar(
        signal_labels,
        img_means,
        yerr=img_stderrs,
        fmt="o-",
        linewidth=1,
        markersize=2,
        color="black",
        label="Mean [SIG]",
    )
    # order legend elements
    handles, labels = plt.gca().get_legend_handles_labels()
    zipped_legends = zip(handles, labels, data_lasts)
    sorted_legends = sorted(zipped_legends, key=lambda l: l[-1], reverse=True)
    handles, labels, _ = zip(*sorted_legends)
    legend = ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small"
    )
    plt.xticks(rotation=90, fontsize="small")
    plt.ylim(bottom=0.0)
    plt.show()
    """

    # Levey-Jennings chart
    signal_labels = [os.path.basename(image) for image in images_img_data.keys()]
    slice_margin = len(files) - 1  # extend slice to either sides
    slice_min = False  # don't show data for incomplete slices
    fit_trend = False  # fit a linear regression model of the mean
    slice_size = 2 * slice_margin + 1
    assert (
        slice_size > 3
    ), "Zero degrees of freedom to estimate the standard deviation from the trend line."
    file_len = len(files)
    xs = range(0, file_len)
    np_nan = np.full(file_len, np.nan)
    for c, chan in enumerate(chans):
        # prepare variables
        trend_stats = {stat: np_nan.copy() for stat in ["vals", "stdevs", "where"]}
        run_stats = {stat: np_nan.copy() for stat in ["slice", "means", "stdevs"]}
        # get image statistics
        signal_means = get_chan_data(images_img_data, chan, "sign_mean")
        signal_stdevs = get_chan_data(images_img_data, chan, "sign_stdev")
        signal_stderrs = get_chan_data(images_img_data, chan, "sign_stderr")
        # get trend statistics
        if fit_trend:
            slope, inter = np.polyfit(xs, signal_means, deg=1)
            trend_stats["vals"] = slope * xs + inter
        else:
            trend_stats["vals"].fill(get_mean(signal_means))
        # get running statistics
        for i, mean in enumerate(signal_means):
            run_stats["slice"] = get_run_slice(signal_means, i, slice_margin, slice_min)
            if not slice_min or run_stats["slice"].size == slice_size:
                run_stats["means"][i] = get_mean(run_stats["slice"])
                run_stats["stdevs"][i] = get_mean(
                    get_run_slice(signal_stdevs, i, slice_margin, slice_min)
                )
                trend_stats["stdevs"][i] = get_stdev(
                    run_stats["slice"],
                    get_mean(
                        get_run_slice(trend_stats["vals"], i, slice_margin, slice_min)
                    ),
                    ddof=3,  # estimated: slope, intercept, and mean
                )
        if not slice_min:
            # fill `stdevs` array with limit values
            trend_stats["where"] = np.where(~np.isnan(trend_stats["stdevs"]))[0]
            if trend_stats["where"].size > 0:  # channel might be sparse with images
                trend_stats["stdevs"][: trend_stats["where"][0]] = trend_stats[
                    "stdevs"
                ][
                    trend_stats["where"][0]
                ]  # extend left
                trend_stats["stdevs"][trend_stats["where"][-1] :] = trend_stats[
                    "stdevs"
                ][
                    trend_stats["where"][-1]
                ]  # extend right
        # plot statistics
        for dist in [2.0, 1.0, -1.0, -2.0]:
            linestyle = (0, (1, 2))
            if abs(dist) == 2.0:
                linestyle = linestyle = (0, (1, 4))
            plt.plot(
                run_stats["means"] + dist * run_stats["stdevs"],
                color="black",
                linewidth=1,
                linestyle=linestyle,
            )
        for dist in [2.0, 1.0]:
            alpha = 0.2
            if abs(dist) == 1.0:
                alpha = 0.1
            plt.fill_between(
                xs,
                trend_stats["vals"] + dist * trend_stats["stdevs"],
                trend_stats["vals"] - dist * trend_stats["stdevs"],
                color="black",
                alpha=alpha,
            )
        plt.fill_between(
            xs,
            trend_stats["vals"] + 1.0 * trend_stats["stdevs"],
            trend_stats["vals"] - 1.0 * trend_stats["stdevs"],
            color="black",
            alpha=0.2,
        )
        plt.plot(trend_stats["vals"], color="black", linewidth=1, linestyle="solid")
        plt.plot(run_stats["means"], color="black", linewidth=1, linestyle="dashed")
        plt.errorbar(
            signal_labels,
            signal_means,
            yerr=signal_stderrs,
            fmt="o-",
            linewidth=1,
            markersize=2,
            color=color_map[c],
            label=chan + " [SIG]",
        )
        legend = plt.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small"
        )
        plt.xticks(rotation=90, fontsize="small")
        plt.ylim(bottom=0.0)
        plt.show()
        plt.clf()
