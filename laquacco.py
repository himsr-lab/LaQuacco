import datetime
import fnmatch
import math
import os
import random
import tifffile
import time
import xmltodict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy as sp


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


def get_chan_img(tiff, channel):
    """Get the channel image from a TIFF dictionary.

    Keyword arguments:
    tiff -- TIFF dictionary
    channel -- channel name
    """
    pixls = np.zeros((tiff["shape"][1:]))  # pre-allocate
    for page in tiff["pages"]:
        chan = get_chan(page)
        if chan == channel:
            page.asarray(out=pixls)  # in-place
    tiff["tiff"].close()
    return pixls


def get_colormap(count):
    """Return a colormap with `counts` number of colors.

    Keyword arguments:
    count  -- number of colors to be generated
    """
    color_points = np.linspace(0, 1, count)
    return [cm.hsv(color_point) for color_point in color_points]

    for i, image in enumerate(images_img_data):
        for chan in chans:
            chans_means[i] += images_img_data[image][chan]["mean"]
            chans_stderrs[i] += images_img_data[image][chan]["stderr"]


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


def get_max(array):
    """Returns the maximum value.

    Keyword arguments:
    array -- Numpy array
    """
    maximum = np.nan
    array = array[~np.isnan(array)].ravel()  # ignore all `np.nan` values
    arrow = pl.from_numpy(array, schema=["pixls"], orient="col")  # cheap
    pixls = arrow["pixls"]
    if len(pixls) > 0:
        maximum = pixls.max()
    return maximum


def get_mean(array):
    """Returns the arithmetic mean.

    Keyword arguments:
    array -- Numpy array
    """
    mean = np.nan
    array = array[~np.isnan(array)].ravel()  # ignore all `np.nan` values
    arrow = pl.from_numpy(array, schema=["pixls"], orient="col")  # cheap
    pixls = arrow["pixls"]
    if len(pixls) > 0:
        mean = pixls.mean()
    return mean


def get_min(array):
    """Returns the maximum value.

    Keyword arguments:
    array -- Numpy array
    """
    minimum = np.nan
    array = array[~np.isnan(array)].ravel()  # ignore all `np.nan` values
    arrow = pl.from_numpy(array, schema=["pixls"], orient="col")  # cheap
    pixls = arrow["pixls"]
    if len(pixls) > 0:
        minimum = pixls.min()
    return minimum


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
    """Calculates basic statistics for a 1-dimensional array: Polars' parallel Rust
    implementation is significantly faster - especially for large Numpy arrays.
    Keyword arguments:
    array -- Numpy array
    """
    array = array[~np.isnan(array)].ravel()  # ignore all `np.nan` values
    arrow = pl.from_numpy(array, schema=["pixls"], orient="col")  # cheap
    sortd = arrow.sort("pixls")  # expensive
    sortd.set_sorted("pixls")  # flag
    pixls = sortd["pixls"]
    size = len(pixls)
    minimum = pixls[0]
    head = pixls.head(math.ceil(0.001 * size)).mean()  # first elements
    mean = pixls.mean()
    stdev = pixls.std()
    stderr = np.power(stdev, 2) / size
    tail = pixls.tail(math.ceil(0.002 * size)).mean()  # last elements
    maximum = pixls[-1]
    return (mean, stdev, stderr, (minimum, maximum), (head, tail))


def get_stderr(array, ddof=1):
    """Calculates the standard error of the arithmetic mean.

    Keyword arguments:
    array -- Numpy array
    size -- number of valid observations
    mean -- arithmetic mean of the observations
    ddof -- number of estimated parameters, reduces degrees of freedom
    """
    stderr = np.nan
    array = array[~np.isnan(array)].ravel()  # ignore all `np.nan` values
    arrow = pl.from_numpy(array, schema=["pixls"], orient="col")  # cheap
    pixls = arrow["pixls"]
    pixls_len = len(pixls)
    if pixls_len - ddof > 0:
        stderr = np.sqrt(pixls.var(ddof=ddof) / pixls_len)  # unreliability measure
    return stderr


def get_stdev(array, ddof=1):
    """Calculates the standard deviation.

    Keyword arguments:
    array -- Numpy array
    size -- number of valid observations
    mean -- arithmetic mean of the observations
    ddof -- number of estimated parameters, reduces degrees of freedom
    """
    stdev = np.nan
    array = array[~np.isnan(array)].ravel()  # ignore all `np.nan` values
    arrow = pl.from_numpy(array, schema=["pixls"], orient="col")  # cheap
    pixls = arrow["pixls"]
    pixls_len = len(pixls)
    if pixls_len - ddof > 0:
        stdev = np.sqrt(pixls.var(ddof=ddof))
    return stdev


def get_tiff(image):
    """Open the TIFF file object and return its hanlde
    plus additional information about the first series
    as a descriptive TIFF dictionary.
    Don't forget to close the file after using it!

    Keyword arguments:
    image -- image file
    """
    # open TIFF file and keep handle open for later use
    tiff = tifffile.TiffFile(image)
    series = tiff.series  # descreasing resolutions
    shape = series[0].shape
    pages = tiff.pages[0 : shape[0]]
    channels = []
    # access all pages of the first series
    for p, page in enumerate(pages):
        # identify channel by name
        chan = get_chan(page)
        # or: by page count
        if not chan:
            chan = str(p)
        channels.append(chan)
    return {
        "tiff": tiff,
        "image": image,
        "shape": shape,
        "pages": pages,
        "channels": channels,
    }


def get_time_left(start=None, current=None, total=None):
    """Return a time in seconds representing the remainder based on
    the durations of the previous iterations (rough estimate).

    Keyword arguments:
    start  -- start time from call to `time.time()`
    current  -- current iteration (positive)
    total -- total iterations
    """
    time_left = None
    now = time.time()
    if now > start:
        if current:
            if current < total:
                time_per_iter = (now - start) / current
                iter_left = total - current
                time_left = iter_left * time_per_iter
            else:
                time_left = 0.0
    return time_left


def get_timestamp(timestamp):
    """Get a timestamp from a corresponding TIFF tag string.

    Keyword arguments:
    timestamp  -- the timestamp string
    """
    return datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")


def score_img_data(tiff, chans_minmax=None):
    """Calculate the C-Scores of image channels.
    The algorithm is similar to an H-Score but scores the different
    intenisty classes with the factors 1, 10, and 10, respectively.

    Keyword arguments:
    tiff -- TIFF dictionary
    chans_minmax -- dictionary with percentile tuples
    """
    img_chans_scores = dict()
    pixls = np.empty((tiff["shape"][1:]))  # pre-allocate
    for page, chan in zip(tiff["pages"], tiff["channels"]):
        page.asarray(out=pixls)  # in-place
        bot, top = (
            chans_minmax[chan]
            if chans_minmax and chan in chans_minmax  # user-defined
            else (get_min(pixls), get_max(pixls))  # automatic
        )
        span = top - bot
        signal = pixls[pixls >= (bot + 0.25 * span)]  # ignore background
        size = signal.size
        counts = [np.nan, np.nan, np.nan]
        counts[0] = np.count_nonzero(signal[signal < (bot + 0.50 * span)])
        counts[2] = np.count_nonzero(signal[signal >= (bot + 0.75 * span)])
        counts[1] = size - counts[0] - counts[2]
        img_chans_scores[chan] = {
            "score_1": 1.0 * counts[0] / size,  # max contribution: + 1
            "score_2": 10.0 * counts[1] / size,  # max contribution: + 10
            "score_3": 100.0 * counts[2] / size,  # max contribution: + 100
        }
    tiff["tiff"].close()
    return img_chans_scores


def stats_img_data(tiff, chan_mins=None):
    """Calculate basic statistics for the image channels.

    Keyword arguments:
    tiff -- TIFF dictionary
    chan_mins -- dictionary with minimum signal values
    """
    img_chans_data = dict()
    pixls = np.empty((tiff["shape"][1:]))  # pre-allocate
    for page, chan in zip(tiff["pages"], tiff["channels"]):
        page.asarray(out=pixls)  # in-place
        # get date and time of acquisition
        img_chans_data["metadata"] = {
            "date_time": get_timestamp(page.tags["DateTime"].value)
        }
        thrld = chan_mins[chan] if chan_mins and chan in chan_mins else 0.0
        signal = pixls[pixls > thrld]  # ignore background
        # get statistics for channel
        img_chans_data[chan] = {}
        if chan_mins:
            mean = get_mean(signal)
            img_chans_data[chan]["mean"] = mean
            img_chans_data[chan]["stdev"] = get_stdev(signal)
            img_chans_data[chan]["stderr"] = get_stderr(signal)
            img_chans_data[chan]["minmax"] = (
                get_min(signal),
                get_max(signal),
            )
        else:
            (
                img_chans_data[chan]["mean"],
                img_chans_data[chan]["stdev"],
                img_chans_data[chan]["stderr"],
                img_chans_data[chan]["minmax"],
                img_chans_data[chan]["headtail"],
            ) = get_stats(signal)
            get_stats(signal)
    tiff["tiff"].close()
    return img_chans_data
