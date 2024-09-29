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
Summary:    Laboratory Quality Control v2.0 (2024-09-28)
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


def get_chan_stats(pixls, chan_limits={}, chan_means=None):
    """Get channel statistics depending on input. Without additional (group)
       statistics, the function will return the max, mean, and min values of
       the input pixels. By providing an additional function argument with
       previously determined (average) statistics, the function returns new
       statistics that depend on the averages.

    Keyword arguments:
    pixls -- two-dimensional NumPy array with channel pixels
    chan_limits -- channel's interval boundaries (lower and upper)
    chan_means -- channel's statistics ("max", "mean", "min")
    """
    result = {}
    # create Polars DataFrame
    frame = pl.from_numpy(pixls.ravel(), schema=["pixls"], orient="col")  # fast
    # set filter on dataframe
    frame = set_chan_interval(frame, chan_limits)
    # prepare computation of statistics
    cbands = chan_means and all(
        [
            stat in chan_means and chan_means[stat] is not None
            for stat in ["max", "mean", "min"]
        ]
    )
    row = pl.col("pixls")
    # get individual channel stats
    query_stats = [
        row.max().alias("max"),
        row.mean().alias("mean"),
        row.min().alias("min"),
    ]
    if cbands:
        # get group channel stats
        signal_range = chan_means["max"] - chan_means["mean"]
        signal_limit_0 = chan_means["mean"]
        signal_limit_1 = chan_means["mean"] + (1.0 / 3.0) * signal_range
        signal_limit_2 = chan_means["mean"] + (2.0 / 3.0) * signal_range
        query_cbands = [
            # band_0: [−∞, signal_limit_0[
            row.filter(row < signal_limit_0).mean().alias("band_0"),
            # band_1: [signal_limit_0, signal_limit_1[
            row.filter((row >= signal_limit_0) & (row < signal_limit_1))
            .mean()
            .alias("band_1"),
            # band_2: [signal_limit_1, signal_limit_2[
            row.filter((row >= signal_limit_1) & (row < signal_limit_2))
            .mean()
            .alias("band_2"),
            # band_3: [signal_limit_2, +∞]
            row.filter(row >= signal_limit_2).mean().alias("band_3"),
        ]
        query_stats.extend(query_cbands)
    stats = frame.select(query_stats)  # execute computation request
    result = {stat: stats.select(stat).item() for stat in stats.columns}
    return result


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
    expo_times = [get_expo_si(expo_time) for expo_time in expo_times]
    return expo_times


def get_expo_si(expo=(1.0, "s")):
    """Convert exposure times with custom exposure unit to SI value/unit tuple.

    Keyword arguments:
    expo -- tuple with exposure time and exposure unit
    """
    time = 1.0
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
    if expo[1] in ome_units:
        time = float(expo[0]) * ome_units[expo[1]]
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


def get_img(file):
    """Open the image as TIFF file object and return its hanlde plus additional metadata
    as a Python dictionary. Don't forget to close the file after using it!

    Keyword arguments:
    file -- image file path
    """
    # open TIFF file and keep handle open for later use
    tiff = tifffile.TiffFile(file)
    xml_meta = get_xml_meta(tiff)
    channels = get_chans(tiff, xml_meta)
    datetimes = get_dates(tiff, xml_meta)
    exposures = get_expos(tiff, xml_meta, channels)
    # return metadata and tiff object
    return {
        "channels": channels,  # channel labels
        "exposures": exposures,  # exposure times
        "datetimes": datetimes,  # acquisition timestamps
        "file": str(file),  # image file path
        "tiff": tiff,  # tiff object
    }


def get_img_chans_stats(image, chans_limits={}, chans_means={}):
    """Calculate basic statistics for the image channels.

    Keyword arguments:
    image -- TIFF metadata and tiff object
    chans_limits -- channels' interval boundaries (lower and upper)
    chans_means  -- mean channels' statistics (min and max)
    """
    img_chans_stats = {}
    # pre-allocate memory for NumPy array
    axes = image["tiff"].pages[0].axes
    shape = image["tiff"].pages[0].shape
    pixls = np.empty((shape[axes.index("Y")], shape[axes.index("X")]))  # pre-allocate
    for page, chan in zip(image["tiff"].pages, image["channels"]):
        page.asarray(out=pixls)  # write in-place
        # prepare specific or generic limits for channel
        chan_limits = chans_limits.get(chan, chans_limits.get("*", None))
        # calculate channel statistics from pixel data
        if chan in chans_means and chans_means[chan] is not None:
            img_chans_stats[chan] = get_chan_stats(
                pixls, chan_limits, chans_means[chan]
            )
        else:
            img_chans_stats[chan] = get_chan_stats(pixls, chan_limits)
    return img_chans_stats


def get_query_results(chan_pixls, query):
    """Using Polars' aggregation function to lazily evaluate the results
       of the (aggregation) functions used. Since we're using a lazy
       dataframe, we have to collect the results to force evaluation.

    Keyword arguments:
    chan_pixls -- Polars lazy dataframe with channel pixels
    query -- list with aggregation functions
    """
    queue = chan_pixls.select(query)  # queue computation request
    stats = queue.collect()  # execute computation request
    results = {stat: value[0] for stat, value in stats.to_dict(as_series=False).items()}
    return results


def get_time_left(start=None, current=None, total=None):
    """Return a time string representing the remainder based on
       the durations of the previous iterations (rough estimate).

    Keyword arguments:
    start  -- start time from call to `time.time()`
    current  -- current iteration
    total -- total iterations
    """
    seconds = 0.0
    if start and current and total:
        now = time.time()  # seconds
        if current < total:
            time_per_iter = (now - start) / current
            iter_left = total - current
            seconds = iter_left * time_per_iter
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


def set_chan_interval(frame, limits={"lower": None, "upper": None}):
    """Filter the channel values for a user-specified interval.
       The limits are closed interval endpoints.

    Keyword arguments:
    frame -- Polars lazy dataframe with channel pixels
    limits -- dictionary with "lower" and "upper" interval limits
    """
    row = pl.col("pixls")
    if limits and any(
        [lim in limits and limits[lim] is not None for lim in ["lower", "upper"]]
    ):
        query_limits = pl.lit(True)  # True literal to start
        if limits.get("lower"):
            lower_condition = row >= limits["lower"]
            query_limits = query_limits & lower_condition
        if limits.get("upper"):
            upper_condition = row <= limits["upper"]
            query_limits = query_limits & upper_condition
        return frame.filter(query_limits)
    else:  # no limits
        return frame
