import fnmatch
import math
import re
import os
import tifffile
import xmltodict

from datetime import datetime
from dateutil import parser


# compile once at module load time
xml_pattern = re.compile(
    r"""
    (?:
        # XML declaration (optional)
        <\?xml\s+version="[\d\.]+"\s*encoding="[\w-]+"\s*\?>
    )?
    (
        # XML opening tag
        <([A-Za-z_][\w\.-]*).*?>
        # XML body tags
        .+?
    )
    (
        # XML closing tag (corresponding)
        </\2\s*>
    )
    """,
    re.DOTALL | re.VERBOSE,
)  # hic sunt dracones 🐉


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
    xml_meta = get_xml_meta(tiff)
    channels = get_chans(tiff, xml_meta)
    datetimes = get_dates(tiff, xml_meta)
    exposures = get_expos(tiff, xml_meta, channels)
    # correct for missing IFDs (Standard BioTools)
    if len(datetimes) < len(channels):
        datetimes = [datetimes[0] for _ in channels]
    if len(exposures) < len(channels):
        exposures = [exposures[0] for _ in channels]
    # return metadata and tiff object
    return {
        "channels": channels,  # channel labels
        "exposures": exposures,  # exposure times
        "datetimes": datetimes,  # acquisition timestamps
        "image": image,  # file path
        "tiff": tiff,  # tiff object
    }


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
    expo_times = [expo_to_si(float(time), str(unit)) for time, unit in expo_times]
    return expo_times


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
    print(f"{tiff['acquisitions']}")
    print(f"{[expo for expo in tiff['exposures']]}")
    print()
    tiff["tiff"].close()
