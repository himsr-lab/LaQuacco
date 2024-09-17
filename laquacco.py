"""
Copyright 2023 The Regents of the University of Colorado

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author:     Christian Rickert <christian.rickert@cuanschutz.edu>
Group:      Human Immune Monitoring Shared Resource (HIMSR)
            University of Colorado, Anschutz Medical Campus

Title:      LaQuacco
Summary:    Laboratory Quality Control v1.0 (2024-09-17)
DOI:        # TODO
URL:        https://github.com/christianrickert/LaQuacco

Description:

# TODO
"""

import fnmatch
import os
import platform
import re
import subprocess
import tempfile
import time
import xmltodict
import math
import numpy as np  # single-threaded function calls, multi-threaded BLAS backends
from datetime import datetime
from dateutil import parser

# set environmental variables before imports
try:  # macOS, Linux
    available_cpu = str(len(os.sched_getaffinity(0)) // 2 or 1)
except AttributeError:  # Windows
    available_cpu = str(os.cpu_count() // 2 or 1)
os.environ["POLARS_MAX_THREADS"] = available_cpu  # used to initialize thread pool
os.environ["TIFFFILE_NUM_THREADS"] = available_cpu  # for de/compressing segments
os.environ["TIFFFILE_NUM_IOTHREADS"] = available_cpu  # for reading file sequences
import polars as pl
import tifffile


# compile regular expression at load time
xml_pattern = re.compile(
    r"""
    (?:
        # XML declaration (optional)
        <\?xml\s+version="[\d\.]+"\s*encoding="[\w-]+"\s*\?>
    )?
    (
        # XML opening tag (mandatory)
        <([A-Za-z_][\w\.-]*).*?>
        # XML body tags
        .+?
    )
    (
        # XML closing tag (mandatory)
        </\2\s*>
    )
    """,
    re.DOTALL | re.VERBOSE,
)  # hic sunt dracones 🐉


def copy_file(src_path="", dst_path=""):
    """Use the operating systems' native commands to copy a (remote) source file
       to a (temporary) destination file. If the destination directory does not exist,
       this function will create a temporary destination directiory first.
       Python's built-in `shutil` file copy can't make use of the maximum transfer speeds
       required for network connections, so we're using system-native commands instead.

    Keyword arguments:
    src_file  -- (remote) source file path
    dst_file  -- (temporary) destination file path
    """
    src_path = os.path.abspath(src_path)
    src_dir, src_file = os.path.split(src_path)
    dst_path = (
        os.path.abspath(dst_path)
        if dst_path
        else os.path.join(tempfile.mkdtemp(), src_file)
    )
    dst_dir, dst_file = os.path.split(dst_path)
    command = {
        "Darwin": ["cp", src_path, os.path.join(dst_dir, src_file)],
        "Linux": ["cp", src_path, os.path.join(dst_dir, src_file)],
        "Windows": [
            "ROBOCOPY",
            src_dir,
            dst_dir,
            src_file,
            "/COMPRESS",
            "/NJH",
            "/NJS",
            "/NP",
        ],
    }.get(platform.system())
    try:
        subprocess.run(command, check=platform.system() != "Windows")
    except subprocess.CalledProcessError as err:
        print(f"Failed to copy file. Error was:\n{err}")
    return os.path.abspath(os.path.join(dst_dir, src_file))


def get_chans(tiff, xml_meta):
    """Get the channel names from a TIFF object.

    Keyword arguments:
    tiff -- the TIFF object
    xml_meta -- XML TIFF metadata
    """
    chans = []
    pages = tiff.series[0].pages
    pages_len = len(pages)
    if xml_meta:
        try:  # OME-TIFF
            chans = [
                chan.get("@Name")
                for chan in xml_meta["OME"]["Image"]["Pixels"]["Channel"]
            ]
        except KeyError:  # OME-variants
            qptiff_ident = "PerkinElmer-QPI-ImageDescription"
            qptiff_metadata = xml_meta.get(qptiff_ident, None)
            if qptiff_metadata:  # PerkinElmer QPTIFF
                for index in range(pages_len):
                    try:  # fluorescence
                        chans.append(get_xml_meta(tiff, index)[qptiff_ident]["Name"])
                    except KeyError:  # brightfield
                        pass  # missing
    if not chans:  # regular TIFF or OME-TIFF without names
        try:
            for page in pages:
                chans.append(page.aspage().tags["PageName"].value)
        except KeyError:  # generic tags
            chans = [f"Channel {channel}" for channel in range(1, pages_len + 1)]
    return chans


def get_chan_data(imgs_chans_data, chan, data, length=1):
    """Returns channel data from image data dictionaries.
       Works across all images to retrieve the channel data.

    Keyword arguments:
    imgs_chans_data -- dictionaries with image data
    chan -- the key determining the channel value
    data -- the key determining the channel data
    length -- length of the data tuple
    """
    chan_data = []
    empty = tuple(None for n in range(0, length)) if length > 1 else None
    for _img, chans_data in imgs_chans_data.items():
        if chan in chans_data and chan not in ["metadata"]:
            chan_data.append(chans_data[chan][data])
        else:  # channel missing in image
            chan_data.append(empty)
    # convert to Numpy array, keep Python datatype
    chan_data = np.array(chan_data, dtype="float")
    chan_data[chan_data is None] = np.nan
    return chan_data


def get_dates(tiff, xml_meta):
    """Get the acquisition timestamps from a TIFF object.

    Keyword arguments:
    tiff -- the TIFF object
    xml_meta -- XML TIFF metadata
    """
    acq = None
    acqs = []
    pages = tiff.series[0].pages
    pages_len = len(pages)
    if xml_meta:
        try:  # OME-TIFF
            acq = xml_meta["OME"]["Image"].get("AcquisitionDate", None)
        except KeyError:  # OME-variants
            pass  # missing
    if not acq:  # regular TIFF or OME-TIFF without timestamp
        try:
            acq = datetime.strptime(
                pages[0].aspage().tags["DateTime"].value, "%Y:%m:%d %H:%M:%S"
            )  # baseline tag
        except KeyError:
            acq = datetime.now()  # generated
    if not isinstance(acq, datetime):
        acq = parser.parse(acq)  # voodoo 🐔
    acqs = [acq for _ in range(pages_len)]
    return acqs


def get_expos(tiff, xml_meta, channels):
    """Get the exposure times from a TIFF object.

    Keyword arguments:
    tiff -- the TIFF object
    xml_meta -- XML TIFF metadata
    channels  -- list of channels
    """
    pages = tiff.series[0].pages
    pages_len = len(pages)
    expo_times = []
    if xml_meta:
        try:  # OME-TIFF
            expo_times = [
                (chan.get("@ExposureTime", None), chan.get("@ExposureTimeUnit", "s"))
                for chan in xml_meta["OME"]["Image"]["Pixels"]["Plane"]
            ]
        except KeyError:  # OME-variants
            qptiff_ident = "PerkinElmer-QPI-ImageDescription"
            qptiff_metadata = xml_meta.get(qptiff_ident, None)
            if qptiff_metadata:  # PerkinElmer QPTIFF
                for index in range(pages_len):
                    expo_times.append(
                        (get_xml_meta(tiff, index)[qptiff_ident]["ExposureTime"], "µs")
                    )  # unit is undefined
    if not expo_times:  # regular TIFF or OME-TIFF without names
        expo_times = [(1.0, "s") for chan in channels]
    expo_times = [get_expo_si(float(time), str(unit)) for time, unit in expo_times]
    return expo_times


def get_expo_si(time=1, unit="s"):
    """Convert exposure times with custom exposure unit to SI value/unit tuple.

    Keyword arguments:
    expo -- tuple with exposure time and exposure unit
    """
    ome_units = {
        "Ys": math.pow(10.0, 24),  # yotta
        "Zs": math.pow(10.0, 21),  # zetta
        "Es": math.pow(10.0, 18),  # exa
        "Ps": math.pow(10.0, 15),  # peta
        "Ts": math.pow(10.0, 12),  # tera
        "Gs": math.pow(10.0, 9),  # giga
        "Ms": math.pow(10.0, 6),  # mega
        "ks": math.pow(10.0, 3),  # kilo
        "hs": math.pow(10.0, 2),  # hecto
        "das": math.pow(10.0, 1),  # deca
        "ds": math.pow(10.0, -1),  # deci
        "cs": math.pow(10.0, -2),  # centi
        "ms": math.pow(10.0, -3),  # milli
        "µs": math.pow(10.0, -6),  # micro
        "ns": math.pow(10.0, -9),  # nano
        "ps": math.pow(10.0, -12),  # pico
        "fs": math.pow(10.0, -15),  # femto
        "as": math.pow(10.0, -18),  # atto
        "zs": math.pow(10.0, -21),  # zepto
        "ys": math.pow(10.0, -24),  # yocto
    }
    if unit in ome_units:
        time = time * ome_units[unit]
    return (time, "s")


def get_files(path="", pat="*", anti="", recurse=False):
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


def get_stats(array, chan_stats=(None, None, None)):
    """Calculates basic statistics for a 1-dimensional array: Polars' parallel Rust
       implementation is significantly faster - especially for large Numpy arrays.

    Keyword arguments:
    array -- Numpy array
    chan_stats -- channel's statistics (mean, min, max)
    """
    chan_mean = chan_stats[0]
    chan_min = chan_stats[1]
    chan_max = chan_stats[2]
    get_bands = chan_mean and chan_min and chan_max
    arrow = pl.from_numpy(array.ravel(), schema=["pixls"], orient="col")  # fast
    if get_bands:
        pixls = arrow.filter(
            pl.col("pixls") >= chan_min
        )  # exclude below-threshold regions
    else:
        pixls = arrow.filter(pl.col("pixls") > chan_min)  # exclude non-signal regions
    total, size = len(arrow), len(pixls)
    mean, minimum, maximum = None, None, None
    band_0, band_1, band_2, band_3 = None, None, None, None
    if size:
        # prepare vectors calculations
        stats = [
            pl.col("pixls").mean().alias("mean"),
            pl.col("pixls").min().alias("min"),
            pl.col("pixls").max().alias("max"),
        ]
        if get_bands:
            # [0---band_0---(mean)---band_1---|---band_2---|---band_3---(max)]
            bands_range = chan_max - chan_mean
            lim_1 = chan_mean + 1.0 / 3.0 * bands_range
            lim_2 = chan_mean + 2.0 / 3.0 * bands_range
            stats.extend(
                [
                    pl.col("pixls")
                    .filter((pl.col("pixls") < chan_mean))
                    .mean()
                    .alias("band_0"),
                    pl.col("pixls")
                    .filter((pl.col("pixls") >= chan_mean) & (pl.col("pixls") < lim_1))
                    .mean()
                    .alias("band_1"),
                    pl.col("pixls")
                    .filter((pl.col("pixls") >= lim_1) & (pl.col("pixls") < lim_2))
                    .mean()
                    .alias("band_2"),
                    pl.col("pixls")
                    .filter((pl.col("pixls") >= lim_2))
                    .mean()
                    .alias("band_3"),
                ]
            )
        # apply vector calculations
        result = pixls.select(stats)  # iterate over pixels only once
        # retrieve vector calculations
        mean = result.select("mean").item()
        minimum = result.select("min").item()
        maximum = result.select("max").item()
        if get_bands:
            band_0 = result.select("band_0").item()
            band_1 = result.select("band_1").item()
            band_2 = result.select("band_2").item()
            band_3 = result.select("band_3").item()
    return (total, size, mean, (minimum, maximum), (band_0, band_1, band_2, band_3))


def get_image(image):
    """Open the image as TIFF file object and return its hanlde plus additional metadata
    as a Python dictionary. Don't forget to close the file after using it!

    Keyword arguments:
    image -- image file path
    """
    # open TIFF file and keep handle open for later use
    tiff = tifffile.TiffFile(image)
    xml_meta = get_xml_meta(tiff)
    channels = get_chans(tiff, xml_meta)
    datetimes = get_dates(tiff, xml_meta)
    exposures = get_expos(tiff, xml_meta, channels)
    # return metadata and tiff object
    return {
        "channels": channels,  # channel labels
        "exposures": exposures,  # exposure times
        "datetimes": datetimes,  # acquisition timestamps
        "image": image,  # image file path
        "tiff": tiff,  # tiff object
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


def stats_img_data(tiff, chans_stats=None):
    """Calculate basic statistics for the image channels.

    Keyword arguments:
    tiff -- TIFF dictionary
    chans_stats  -- channels' statistics (mean, min, and max)
    """
    img_chans_data = dict()
    pixls = np.empty((tiff["shape"][1:]))  # pre-allocate
    for page, chan in zip(tiff["pages"], tiff["channels"]):
        page.asarray(out=pixls)  # in-place
        # get date and time of acquisition
        img_chans_data["metadata"] = {
            "date_time": get_timestamp(page.tags["DateTime"].value)
        }
        if not chans_stats or chan not in chans_stats:
            chans_stats = {chan: (None, 0.0, None)}
        # get statistics for channel
        img_chans_data[chan] = {}
        (
            img_chans_data[chan]["total"],
            img_chans_data[chan]["size"],
            img_chans_data[chan]["mean"],
            img_chans_data[chan]["minmax"],
            img_chans_data[chan]["bands"],
        ) = get_stats(pixls, chans_stats[chan])
    tiff["tiff"].close()
    return img_chans_data


def get_xml_meta(tiff, page=0):
    """Get OME metadata from `ImageDescription` TIFF tag and return a Python dictionary.
    We're not relying on `tifffile` and its `ome_metadata` attribute, because it is too
    restrictive for OME-TIFF variants such as PerkinElmer's QPTIFF image files.

    Keyword arguments:
    tiff -- the TIFF object
    page -- the series (IFDs) index
    """
    xml_metadata = None
    img_dscr = tiff.pages[page].aspage().tags.get("ImageDescription", None)
    if img_dscr:  # TIFF comment contains data
        xml_match = re.search(xml_pattern, img_dscr.value)
        if xml_match:  # TIFF comment matches XML pattern
            xml_metadata = xmltodict.parse(
                img_dscr.value[xml_match.start() : xml_match.end()]
            )
    return xml_metadata


if __name__ == "__main__":
    # Execute when the module is not initialized from an import statement.
    files = get_files(path="./tests/Akoya Biosciences", pat="*.tif")
    copies = [copy_file(file) for file in files]
    images = [get_image(copy) for copy in copies]
    for image in images:
        print(image, "\n")
        image["tiff"].close()
    for copy in copies:
        os.remove(copy)
        print(f"Removed: {copy}")
