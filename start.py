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
    its optimal value that maximizes the log-likelihood function.
    The data needs to be positive (above zero).

    Keyword arguments:
    array  -- the untransformed Numpy array
    lambda  -- the transformation parameter
    """
    if lmbda:
        boxcox = sp.stats.boxcox(array[array > 0], lmbda=lmbda, alpha=None)
        maxlog = None
    else:
        boxcox, maxlog = sp.stats.boxcox(array[array > 0], lmbda=lmbda, alpha=None)
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
    chan_data[chans_data == None] = np.nan
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
        if slice_min and slice.size < (1 + 2 * slice_margin):
            slice = np.array([])
    else:
        slice = np.array([])
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
    mean = np.mean(array)
    stdev = np.std(array, ddof=1)  # estimating arithmetic mean
    stderr = get_stderr(array)
    boxplt = None  # calculate_boxplot(array)
    return (mean, stdev, stderr, boxplt)


def get_stderr(array, mean=None, pop=False, out=False):
    """Calculates the sample standard error of the arithmetic mean.
    If we have observed the whole population (all possible samples),
    then we would have as many degrees of freedom as observed samples.
    However, by default we only observe a fraction of all possilbe samples,
    so we lose one degree of freedom (n-k) - for estimating the arithmetic mean.

    Keyword arguments:
    array -- Numpy array
    mean -- arithmetic mean of the samples
    pop -- the samples represent the population
    """
    if array.size:
        variance = np.nan
        stderr = np.nan
        if not mean:
            mean = np.mean(array)
        sums_squared = np.sum(np.power(array - mean, 2))
        if pop and array.size > 0:  # d.f. = n
            variance = sums_squared / array.size
        elif not pop and array.size > 1:  # d.f. = n - k
            variance = sums_squared / (array.size - 1)
        stderr = np.sqrt(variance / array.size)  # unreliability measure
        return stderr
    else:
        return np.nan


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
            # get pixel data as flattend Numpy array
            pixls = page.asarray().flatten()
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
                img_chans_data[chan]["sign_min"] = pixls_sign_min
                # get basic statistics for signal
                sign_mean, sign_stdev, sign_stderr, sign_boxplt = get_stats(
                    pixls[pixls >= pixls_sign_min]
                )
                img_chans_data[chan]["sign_mean"] = sign_mean
                img_chans_data[chan]["sign_stdev"] = sign_stdev
                img_chans_data[chan]["sign_stderr"] = sign_stderr
                # img_chans_data[chan]["sign_bxplt"] = sign_boxplt
                # get basic statistics for background
                bckg_mean, bckg_stdev, bckg_stderr, bckg_boxplt = get_stats(
                    pixls[pixls < pixls_sign_min]
                )
                img_chans_data[chan]["bckg_mean"] = bckg_mean
                img_chans_data[chan]["bckg_stdev"] = bckg_stdev
                img_chans_data[chan]["bckg_stderr"] = bckg_stderr
                # img_chans_data[chan]["bckg_bxplt"] = bckg_boxplt
            else:  # lambda not yet determined
                norms, chan_lmbda = boxcox_transform(pixls)
                img_chans_data[chan]["chan_lmbda"] = chan_lmbda
        return (image, img_chans_data)


processes = multiprocessing.cpu_count() - 2  # // 2 or 1  # concurrent workers


# main program
if __name__ == "__main__":
    # safe import of main module avoids spawning multiple processes simultaneously
    if platform.system() == "Windows":
        multiprocessing.freeze_support()  # required by 'multiprocessing'
    # get a list of all image files
    files = sorted(
        get_files(
            # path=r"/Users/christianrickert/Desktop/Polaris",
            path=r"/Users/christianrickert/Desktop/MIBI/UCD158/raw",
            pat="*.tiff",
            anti="",
        ),
        key=str.lower,
    )
    # sample experimental image data
    try:
        samples = sorted(get_samples(population=files, perc=20), key=str.lower)
        sample_args = [(sample, None) for sample in samples]
    except ValueError:
        print("Could not draw samples from experimental population.")
        sys.exit(1)
    # analyze the sample
    with multiprocessing.Pool(processes) as pool:
        sample_results = pool.starmap(read_img_data, sample_args)
        pool.close()  # wait for worker tasks to complete
        pool.join()  # wait for worker process to exit

    # print(samples_img_data)
    samples_img_data = {sample: img_data for (sample, img_data) in sample_results}

    chans_set = set()  # avoid duplicate entries
    for img_data in samples_img_data.values():
        for chan in img_data:
            if chan not in ["metadata"]:
                chans_set.add(chan)
    chans = sorted(chans_set, key=str.lower)
    print(chans)

    # prepare colormap
    color_map = get_colormap(len(chans))

    # prepare lambdas for power transform
    chan_lmbdas = {}
    for chan in chans:
        chan_data = get_chan_data(samples_img_data, chan, "chan_lmbda")
        chan_mean = np.nanmean(chan_data)
        chan_lmbdas[chan] = chan_mean
    # print(chan_lmbdas)

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
    plt.ylim(bottom=0.0)
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
        img_means.append(np.nanmean(get_img_data(images_img_data, img, "sign_mean")))
        img_stderrs.append(
            np.nanmean(get_img_data(images_img_data, img, "sign_stderr"))
        )
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
    slice_margin = 3
    slice_min = True
    for c, chan in enumerate(chans):
        # get image statistics
        signal_means = get_chan_data(images_img_data, chan, "sign_mean")
        signal_stdevs = get_chan_data(images_img_data, chan, "sign_stdev")
        signal_stderrs = get_chan_data(images_img_data, chan, "sign_stderr")
        # get running statistics
        np_nan = np.array([np.nan for n in range(0, signal_means.size)])
        run_means = np_nan.copy()
        run_up2stdevs = np_nan.copy()
        run_up1stdevs = np_nan.copy()
        run_dwn1stdevs = np_nan.copy()
        run_dwn2stdevs = np_nan.copy()
        stdev = np.nan
        # get trend statistics
        xs = range(0, len(signal_means))
        slope, inter = np.polyfit(xs, signal_means, deg=1)
        trends = slope * xs + inter
        trend_stderrs = get_chan_data(images_img_data, chan, "sign_stderr")
        for i, mean in enumerate(signal_means):
            slice_means = get_run_slice(signal_means, i, slice_margin, slice_min)
            if slice_means.size > 0:
                run_means[i] = np.nanmean(slice_means)
                slice_stdevs = get_run_slice(signal_stdevs, i, slice_margin, slice_min)
                if slice_stdevs.size > 0:
                    stdev = np.nanmean(slice_stdevs)
                    run_up2stdevs[i] = run_means[i] + 2 * stdev
                    run_up1stdevs[i] = run_means[i] + 1 * stdev
                    run_dwn1stdevs[i] = run_means[i] - 1 * stdev
                    run_dwn2stdevs[i] = run_means[i] - 2 * stdev
                trend_stderrs[i] = get_stderr(
                    slice_means,
                    np.nanmean(get_run_slice(trends, i, slice_margin, slice_min)),
                )
        trend_up1stderrs = trends + trend_stderrs
        trend_dwn1stderrs = trends - trend_stderrs
        # plot statistics
        plt.fill_between(
            xs, trend_up1stderrs, trend_dwn1stderrs, color="black", alpha=0.2
        )
        plt.plot(trends, color="black", linewidth=1, linestyle="solid")
        plt.plot(run_up2stdevs, color="black", linewidth=1, linestyle=(0, (1, 4)))
        plt.plot(run_up1stdevs, color="black", linewidth=1, linestyle=(0, (1, 2)))
        plt.plot(run_means, color="black", linewidth=1, linestyle="dashed")
        plt.plot(run_dwn1stdevs, color="black", linewidth=1, linestyle=(0, (1, 2)))
        plt.plot(run_dwn2stdevs, color="black", linewidth=1, linestyle=(0, (1, 4)))
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

    """
    # code snippets
    # plt.yscale("log")
    #    yticks = [0, 0.1]
    # ymax = np.nanmax(means)
    #    while max(yticks) < np.max(means):
    #        yticks.append(yticks[-1] * 10.0)
    #    plt.gca().set_yticks(yticks)  # Set the ticks positions
    #    plt.gca().set_yticklabels([str(ytick) for ytick in yticks])
    plt.ylim(bottom=0.0)
    plt.show()

    for c, chan in enumerate(chans):
        # get statistics summary
        signal_data = get_chan_data(images_img_data, chan, "sign_mean")
        # signal_mean = np.nanmean(signal_data)
        # signal_stds = get_chan_data(images_img_data, chan, "sign_stdev")
        # signal_errs = get_chan_data(images_img_data, chan, "sign_stderr")
        # signal_std = np.nanmean(signal_stds)
        # background_data = get_chan_data(images_img_data, chan, "bckg_mean")
        # background_mean = np.nanmean(background_data)
        # background_stds = get_chan_data(images_img_data, chan, "bckg_stdev")
        # background_errs = get_chan_data(images_img_data, chan, "bckg_stderr")
        # background_std = np.nanmean(background_stds)

        # if signal_mean - 2 * signal_std > 0:
        #     ax.axhline(
        #         y=signal_mean - 2 * signal_std,
        #         color=color_map[c],
        #         linestyle="dotted",
        #     )
        # if signal_mean - signal_std > 0:
        #     ax.axhline(
        #         y=signal_mean - signal_std, color=color_map[c], linestyle="dashed"
        #     )
        # ax.axhline(y=signal_mean, color=color_map[c], linestyle="dashdot")
        # ax.axhline(
        #     y=signal_mean + signal_std, color=color_map[c], linestyle="dashed"
        # )
        # ax.axhline(
        #     y=signal_mean + 2 * signal_std, color=color_map[c], linestyle="dotted"
        # )
        #
        # last_values.append(signal_data[-1])
        # ax.errorbar(
        #    range(len(signal_data)),
        #    signal_data,
        #    yerr=signal_errs,
        #    fmt="o-",
        #    color=color_map[c],
        #    label=chan + " [SIG]",
        # )
        data_means.append(signal_data)
        data_norms.append(
            boxcox_transform(np.array(signal_data), lmbda=chan_lmbdas[chan])[0]
        )

        # plot statistics summary
        # if background_mean - 2 * background_std > 0:
        #     ax.axhline(
        #         y=background_mean - 2 * background_std,
        #         color=color_map[c],
        #         linestyle="dotted",
        #     )
        # if background_mean - background_std > 0:
        #     ax.axhline(
        #         y=background_mean - background_std, color=color_map[c], linestyle="dashed"
        #     )
        # ax.axhline(y=background_mean, color=color_map[c], linestyle="dashdot")
        # ax.axhline(
        #     y=background_mean + background_std, color=color_map[c], linestyle="dashed"
        # )
        # ax.axhline(
        #     y=background_mean + 2 * background_std, color=color_map[c], linestyle="dotted"
        # )
        # last_values.append(background_data[-1])
        # ax.errorbar(
        #    range(len(background_data)),
        #    background_data,
        #    yerr=background_errs,
        #    fmt=".--",
        #    color=color_map[c],
        #    label=chan + " [BGR]",
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

        # sorted_samples = dict(
        #    sorted(
        #        images_img_data.items(), key=lambda v: v[1]["metadata"]["date_time"]
        #    )
        # )
        # for sorted_sample in sorted_samples:
        #    print(
        #        f"{sorted_sample} -> {images_img_data[sorted_sample]['metadata']['date_time']}"
        #    )
    # plt.yscale("log")
    #    yticks = [0, 0.1]
    # ymax = np.nanmax(means)
    #    while max(yticks) < np.max(means):
    #        yticks.append(yticks[-1] * 10.0)
    #    plt.gca().set_yticks(yticks)  # Set the ticks positions
    #    plt.gca().set_yticklabels([str(ytick) for ytick in yticks])
    plt.ylim(bottom=0.0)
    plt.show()
    """
