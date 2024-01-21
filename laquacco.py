import datetime
import fnmatch
import os
import tifffile
import time
import xmltodict
import matplotlib.cm as cm
import numpy as np
import polars as pl


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
    chan_data[chan_data is None] = np.nan
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
    img_data[img_data is None] = np.nan
    return img_data


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


def get_stats(array, chan_minmax=(None, None)):
    """Calculates basic statistics for a 1-dimensional array: Polars' parallel Rust
    implementation is significantly faster - especially for large Numpy arrays.

    Keyword arguments:
    array -- Numpy array
    chan_min  -- signal threshold (maximum value of background)
    """
    arrow = pl.from_numpy(array.ravel(), schema=["pixls"], orient="col")  # fast
    pixls = arrow.filter(pl.col('pixls') > chan_minmax[0])  # exclude background
    size = len(pixls)
    if chan_minmax[1]:  # get stats and score
        coeff = 0.21544346900318836
        grate = 0.02666666666666667
        result = (  # iterate over pixels only once
            pixls.select([
                pl.col("pixls").mean().alias("mean"),
                pl.col("pixls").std().alias("stdev"),
                pl.col("pixls").min().alias("min"),
                pl.col("pixls").max().alias("max"),
                (coeff * 10.0 ** (grate * pl.col("pixls") / chan_minmax[1]))\
                .sum().alias("score")  # slow
                # Pixels are weighed by their values in relation to the max value:
                # Examples: 0%: +0.0, 25%: +1.0, 50%: +4.6, 75%: +21.5, 100%: +100
            ])
        )
        mean = result.select("mean").item()
        stdev = result.select("stdev").item()
        stderr = np.sqrt(np.power(result.select("stdev").item(), 2.0) / size)
        score = result.select("score").item()
    else:  #  get minimal score
        result = (
            pixls.select([
                pl.col("pixls").min().alias("min"),
                pl.col("pixls").max().alias("max"),
            ])
        )
        mean = None
        stdev = None
        stderr = None
        score = None
    minimum = result.select("min").item()
    maximum = result.select("max").item()
    return (size, mean, stdev, stderr, (minimum, maximum), score)


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
    """Return a time string representing the remainder based on
    the durations of the previous iterations (rough estimate).

    Keyword arguments:
    start  -- start time from call to `time.time()`
    current  -- current iteration (positive)
    total -- total iterations
    """
    seconds = 0.0
    now = time.time()  # seconds
    if now > start:
        if current:
            if current < total:
                time_per_iter = (now - start) / current
                iter_left = total - current
                seconds = iter_left * time_per_iter
            else:
                seconds = 0.0
    # create format string
    time_str = ""
    sec_min = 60.0
    sec_hour = 60.0 * sec_min
    sec_day = 24.0 * sec_hour
    days = seconds // sec_day
    seconds %= sec_day
    hours = seconds // sec_hour
    seconds %= sec_hour
    minutes = seconds // sec_min
    seconds %= sec_min
    if days > 0.0:
        time_str += f"{round(days)}d "
    if hours > 0.0:
        time_str += f"{round(hours)}h "
    if minutes > 0.0:
        time_str += f"{round(minutes)}m "
    if seconds >= 0.0:
        time_str += f"{round(seconds)}s"
    return time_str.strip()


def get_timestamp(timestamp):
    """Get a timestamp from a corresponding TIFF tag string.

    Keyword arguments:
    timestamp  -- the timestamp string
    """
    return datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")


def stats_img_data(tiff, chans_minmax=None):
    """Calculate basic statistics for the image channels.

    Keyword arguments:
    tiff -- TIFF dictionary
    chans_min  -- signal thresholds (maximum values of backgrounds)
    """
    img_chans_data = dict()
    pixls = np.empty((tiff["shape"][1:]))  # pre-allocate
    for page, chan in zip(tiff["pages"], tiff["channels"]):
        page.asarray(out=pixls)  # in-place
        # get date and time of acquisition
        img_chans_data["metadata"] = {
            "date_time": get_timestamp(page.tags["DateTime"].value)
        }
        if not chans_minmax or chan not in chans_minmax:
            chans_minmax = {chan: (0.0, None)}
        # get statistics for channel
        img_chans_data[chan] = {}
        (
            img_chans_data[chan]["size"],
            img_chans_data[chan]["mean"],
            img_chans_data[chan]["stdev"],
            img_chans_data[chan]["stderr"],
            img_chans_data[chan]["minmax"],
            img_chans_data[chan]["score"],
        ) = get_stats(pixls, chans_minmax[chan])
    tiff["tiff"].close()
    return img_chans_data

