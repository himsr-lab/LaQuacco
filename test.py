import fnmatch
import math
import os
import tifffile
import xmltodict
import dateutil
from datetime import datetime


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
    ome_meta = get_ome_meta(tiff)
    channels = get_chans(tiff, ome_meta)
    date_time = get_date_time(tiff, ome_meta)
    expo_times = get_expo_times(tiff, ome_meta, channels)
    series = tiff.series  # image file directories (IFD)
    shape = series[0].shape
    pages = tiff.pages[0 : shape[0]]
    return {
        "channels": channels,  # labels
        "date_time": date_time,  # timestamp
        "expo_times": expo_times,  # acquisition
        "image": image,  # file path
        "ome_meta": ome_meta,  # OME TIFF metadata
        "pages": pages,  # data pages
        "shape": shape,  # dimensions
        "tiff": tiff,  # tiff object
    }


def get_date_time(tiff, ome_meta):
    """Get a datetime from the corresponding TIFF metadata.

    Keyword arguments:
    tiff -- the TIFF object
    ome_meta -- OME TIFF metadata
    """
    date_time = None
    if ome_meta:
        try:  # OME-TIFF
            date_time = ome_meta["OME"]["Image"].get("AcquisitionDate", None)
        except KeyError:  # OME-variants
            qptiff_ident = "PerkinElmer-QPI-ImageDescription"
            qptiff_metadata = ome_meta.get(qptiff_ident, None)
            if qptiff_metadata:  # PerkinElmer QPTIFF
                try:  # fluorescence
                    filter = ome_meta[qptiff_ident]["Responsivity"].get("Filter")
                    if filter:  # Akoya Vectra 3
                        if isinstance(filter, list):
                            filter = filter[0]
                        date_time = filter.get("Date")
                    band = ome_meta[qptiff_ident]["Responsivity"].get("Band")
                    if band:  # Akoya Vectra Polaris/PhenoImager HT (FOVs)
                        if isinstance(band, list):
                            band = band[0]
                        date_time = band.get("Date")
                except KeyError:  # brightfield
                    pass  # missing
    if not date_time:  # regular TIFF or OME-TIFF without `Date` field
        date_time = tiff.series[0].pages[0].aspage().tags.get("DateTime")
        if date_time:  # with baseline tag
            date_time = datetime.strptime(str(date_time.value), "%Y:%m:%d %H:%M:%S")
        else:  # without baseline tag
            date_time = datetime.now()
    if not isinstance(date_time, datetime):
        date_time = dateutil.parser.parse(date_time)  # voodoo
    return date_time


def expo_to_si(time=1, unit="s"):
    """Convert exposure times with custom exposure unit to SI values.

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


def get_expo_times(tiff, ome_meta, channels):
    """Get the exposure times from a TIFF object.

    Keyword arguments:
    tiff -- the TIFF object
    ome_meta -- OME TIFF metadata
    channels  -- list of channels
    """
    expo_times = []
    pages = tiff.series[0].pages
    if ome_meta:
        try:  # OME-TIFF
            expo_times = [
                (chan.get("@ExposureTime", None), chan.get("@ExposureTimeUnit", "s"))
                for chan in ome_meta["OME"]["Image"]["Pixels"]["Plane"]
            ]
        except KeyError:  # OME-variants
            qptiff_ident = "PerkinElmer-QPI-ImageDescription"
            qptiff_metadata = ome_meta.get(qptiff_ident, None)
            if qptiff_metadata:  # PerkinElmer QPTIFF
                for index, page in enumerate(pages):
                    expo_times.append(
                        (get_ome_meta(tiff, index)[qptiff_ident]["ExposureTime"], "µs")
                    )  # unit is undefined
    if not expo_times:  # regular TIFF or OME-TIFF without names
        expo_times = [(1.0, "s") for chan in channels]
    expo_times = [expo_to_si(float(time), str(unit)) for time, unit in expo_times]
    return expo_times


def get_ome_meta(tiff, page=0):
    """Get OME metadata from `ImageDescription` TIFF tag and
    return a Python dictionary.

    Keyword arguments:
    tiff -- the TIFF object
    page -- the series (IFDs) index
    """
    ome_metadata = None
    img_dscr = tiff.pages[page].tags.get("ImageDescription", None)
    if img_dscr and img_dscr.value.startswith("<?xml"):
        ome_metadata = xmltodict.parse(img_dscr.value)
    return ome_metadata


def get_chans(tiff, ome_meta):
    """Get the channel names from a TIFF object.

    Keyword arguments:
    tiff -- the TIFF object
    ome_meta -- OME TIFF metadata
    """
    chans = []
    pages = tiff.series[0].pages
    if ome_meta:
        try:  # OME-TIFF
            chans = [
                chan.get("@Name")
                for chan in ome_meta["OME"]["Image"]["Pixels"]["Channel"]
            ]
        except KeyError:  # OME-variants
            qptiff_ident = "PerkinElmer-QPI-ImageDescription"
            qptiff_metadata = ome_meta.get(qptiff_ident, None)
            if qptiff_metadata:  # PerkinElmer QPTIFF
                for index, page in enumerate(pages):
                    try:  # fluorescence
                        chans.append(get_ome_meta(tiff, index)[qptiff_ident]["Name"])
                    except KeyError:  # brightfield
                        pass  # RGB
    if not chans:  # regular TIFF or OME-TIFF without names
        try:
            for page in pages:
                chans.append(page.aspage().tags["PageName"].value)
        except KeyError:  # generic tags
            chans = [f"Channel {channel}" for channel in range(1, len(pages) + 1)]
    return chans


def get_files(path="", pat=None, anti=None, recurse=True):
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


FILES = []
for extension in ["*.tif", "*.tiff", "*.qptiff"]:
    FILES.append(get_files(path="./tests", pat=extension, anti=""))
FILES = [file for sublist in FILES for file in sublist]

for index, FILE in enumerate(FILES):
    path, file = os.path.split(FILE)
    print(f"{index}: {os.path.split(path)[-1]}/{file}:", flush=True)
    tiff = get_tiff(FILE)
    print(f"{[chan for chan in tiff['channels']]}")
    print(f"{tiff['date_time']}")
    print(f"{[expo for expo in tiff['expo_times']]}")
    print()
    tiff["tiff"].close()
