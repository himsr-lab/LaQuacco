import datetime
import fnmatch
import math
import os
import random
import tifffile
import xmltodict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def boxcox_transform(array, lmbda=None):
    """Power-transforms a Numpy array with the `numpy.stats.boxcox` function.
    If the value of `lambda` is not given (None), the fuction will determine
    the optimal value that maximizes the log-likelihood function.
    The input data needs to be positive (above zero).

    Keyword arguments:
    array  -- the untransformed Numpy array
    lambda  -- the transformation parameter
    """
    array = array[array > 0.0]  # mask missing observations
    if lmbda:
        boxcox = sp.stats.boxcox(array, lmbda=lmbda, alpha=None)
        maxlog = None
    else:
        boxcox, maxlog = sp.stats.boxcox(array, lmbda=None, alpha=None)
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


def get_files(path="", pat=None, anti=None, recurse=False):
    """Iterate through all files in a directory structure and
    return a list of matching files.

    Keyword arguments:
    path -- the path to a directory containing files (default "")
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


def get_mean(array, size=None):
    """Returns the arithmetic mean.
    Numpy produces a `RuntimeWarning` when all elements of the array are
    `np.nan` values. Let's just return `np.nan` without warning.

    Keyword arguments:
    array -- Numpy array
    size -- number of valid observations
    """
    mean = np.nan
    if not size:
        size = np.count_nonzero(~np.isnan(array))
    if size > 0:
        mean = np.nanmean(array)
    return mean


def get_run_slice(array, index, slice_margin):
    """Returns the slice of an array centered at the index
     with a margin of elements included before and after.

    Keyword arguments:
    array -- Numpy array
    index  -- center position of the slice
    margin  -- element count before and after index
    """
    slice = np.empty(0)
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
    """Calculates basic statistics for a 1-dimensional array.

    Keyword arguments:
    array -- Numpy array
    """
    stats = sp.stats.describe(array, ddof=1, nan_policy="omit")
    mean = stats.mean
    stdev = np.sqrt(stats.variance)
    stderr = get_stderr(array, stats.nobs, mean)
    minmax = (stats.minmax[0], stats.minmax[1])
    nobs = stats.nobs
    return (mean, stdev, stderr, minmax, nobs)


def get_stderr(array, size=None, mean=None, ddof=1):
    """Calculates the standard error of the arithmetic mean.

    Keyword arguments:
    array -- Numpy array
    size -- number of valid observations
    mean -- arithmetic mean of the observations
    ddof -- number of estimated parameters, reduces degrees of freedom
    """
    stderr = np.nan
    if not size:
        size = np.count_nonzero(~np.isnan(array))
    if size > 0:
        if not mean:
            mean = get_mean(array, size)
        stderr = np.sqrt(
            get_var(array, size, mean, ddof) / size
        )  # unreliability measure
    return stderr


def get_stdev(array, size=None, mean=None, ddof=1):
    """Calculates the standard deviation.

    Keyword arguments:
    array -- Numpy array
    size -- number of valid observations
    mean -- arithmetic mean of the observations
    ddof -- number of estimated parameters, reduces degrees of freedom
    """
    stdev = np.nan
    if not size:
        size = np.count_nonzero(~np.isnan(array))
    if size > 0:
        if not mean:
            mean = get_mean(array, size)
        stdev = np.sqrt(get_var(array, size, mean, ddof))
    return stdev


def get_var(array, size=None, mean=None, ddof=1):
    """Calculates the variance - the variability of the data.
    If we have observed the whole population (all possible samples),
    then we would have as many degrees of freedom as observed samples.
    However, by default we only observe a fraction of all possilbe samples,
    so we lose one degree of freedom (n-k) - for estimating the arithmetic mean.

    Keyword arguments:
    array -- Numpy array
    size -- number of valid observations
    mean -- arithmetic mean of the observations
    ddof -- number of estimated parameters, reduces degrees of freedom
    """
    variance = np.nan
    if not size:
        size = np.count_nonzero(~np.isnan(array))
    if size > 0:
        if not mean:
            mean = get_mean(array, size)
        sums_squared = np.nansum(np.power(array - mean, 2))
        degrs_freedom = size - ddof
        if degrs_freedom > 0:
            variance = sums_squared / degrs_freedom
    return variance


def get_timestamp(timestamp):
    """Get a timestamp from a corresponding TIFF tag string.

    Keyword arguments:
    timestamp  -- the timestamp string
    """
    return datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")


def read_img_data(image, chan_thrlds=None):
    """Calculate the mean values and standard deviations for each of the image channels.

    Keyword arguments:
    images -- image file list
    """
    img_chans_data = dict()
    img_name = os.path.basename(image)
    if chan_thrslds:
        print(f"IMAGE: {img_name}", flush=True)
        get_stats = True
    else:
        print(f"SAMPLE: {img_name}", flush=True)
        get_stats = False
    # open TIFF file to extract image information
    with tifffile.TiffFile(image) as tif:
        date_time = None
        series = tif.series
        pages, rows, columns = series[0].shape
        pixls = np.empty((rows, columns))
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
            if get_stats:
                # get date and time of acquisition
                if not date_time:
                    date_time = get_timestamp(page.tags["DateTime"].value)
                    date_time = img_chans_data["metadata"] = {"date_time": date_time}
                assert p < len(chan_thrlds), "Signal threshold missing for channel."
                # get basic statistics for signal
                (
                    img_chans_data[chan]["sign_mean"],
                    img_chans_data[chan]["sign_stdev"],
                    img_chans_data[chan]["sign_stderr"],
                    img_chans_data[chan]["sign_minmax"],
                    img_chans_data[chan]["sign_nobs"],
                ) = get_stats(pixls[pixls > chan_thrlds[chan]])
            else:  # lambdas and thresholds not yet determined
                norms, lmbda = boxcox_transform(pixls)
                img_chans_data[chan]["chan_lmbda"] = lmbda
                # identify background as bottom outliers from normally distributed signal
                boxplt_data = calculate_boxplot(norms)
                norms_sign_thr = boxplt_data["whiskers"][0].get_ydata()[1]
                pixls_sign_thr = sp.special.inv_boxcox(norms_sign_thr, lmbda)
                if np.isnan(pixls_sign_thr):  # bottom whisker missing
                    pixls_sign_thr = 0.0
                img_chans_data[chan]["chan_thrld"] = pixls_sign_thr
        return (image, img_chans_data)
