import tifffile
import xmltodict
import os
import fnmatch
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
    # expo_times = get_expotimes(tiff)
    series = tiff.series  # image file directories (IFD)
    shape = series[0].shape
    pages = tiff.pages[0 : shape[0]]
    return {
        "channels": channels,  # labels
        "date_time": date_time,  # timestamp
        # "expo_times": expo_times,  # acquisition
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
                date_time = ome_meta[qptiff_ident]["Responsivity"]["Band"][0]["Date"]
    if not date_time:  # regular TIFF or OME-TIFF without timestamp
        page = tiff.series[0].pages[0].aspage()
        try:
            date_time = page.tags["DateTime"].value
        except KeyError:
            date_time = str(datetime.now())

    # if not chans:  # regular TIFF or OME-TIFF without timestamp
    #    try:
    #        pass
    # date_time = tiff["pages"].
    #    try:
    #        for page in pages:
    #            chans.append(page.aspage().tags["PageName"].value)
    #    except KeyError:  # generic tags
    #        chans = [f"Channel {channel}" for channel in range(1, len(pages) + 1)]

    # ome_metadata = tiff.ome_metadata
    # if ome_metadata:  # OME-TIFF
    #    ome_dict = xmltodict.parse(ome_metadata)
    #    date_time = ome_dict["OME"]["Image"].get("AcquisitionDate", None)
    #    try:  # 1989 C standard
    #        date_time = datetime.strptime(date_time, "%Y-%m-%dT%H:%M:%S")
    #    except ValueError:  # others
    #        date_time = datetime.strptime(
    #            date_time[:26] + date_time[27:], "%Y-%m-%dT%H:%M:%S.%f%z"
    #        )
    # else:  # regular TIFF
    #    pages = tiff.series[0].pages
    #    try:  # baseline tags
    #        date_time = datetime.strptime(
    #            pages[0].tags.get("DateTime", None).value, "%Y:%m:%d %H:%M:%S"
    #        )
    #    except ValueError:
    #        pass
    # if not date_time:
    #    date_time = datetime.strptime("1900:00:00T00:00:00", "%Y:%m:%d %H:%M:%S")
    return date_time


# def get_expotimes(tiff):
#    """Get the exposure times from a TIFF object.
#
#    Keyword arguments:
#    tiff -- the TIFF object
#    """
#    expotimes = []
#    ome_metadata = tiff.ome_metadata
#    if ome_metadata:  # OME-TIFF
#        ome_dict = xmltodict.parse(ome_metadata)
#        plane = ome_dict["OME"]["Image"]["Pixels"].get("Plane", None)
#        if plane:
#            expotimes = [plan["@ExposureTime"] for plan in plane]
#        else:
#            sizec = int(ome_dict["OME"]["Image"]["Pixels"].get("@SizeC", None))
#            expotimes = [1.0 for channel in range(1, sizec)]
#    else:  # regular TIFF
#        pages = tiff.series[0].pages
#        expotimes = [1.0 for channel in range(1, len(pages) + 1)]
#    return expotimes
#


def get_ome_meta(tiff, page=0):
    """Get OME metadata from `ImageDescription` TIFF tag and
    return a Python dictionary.

    Keyword arguments:
    tiff -- the TIFF object
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
                    chans.append(get_ome_meta(tiff, index)[qptiff_ident]["Name"])
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
for extension in ["*.tif", "*.tiff"]:
    FILES.append(get_files(path="./tests", pat=extension, anti=""))
FILES = [file for sublist in FILES for file in sublist]

for index, FILE in enumerate(FILES):
    tiff = get_tiff(FILE)
    path, file = os.path.split(FILE)
    print(f"{index}: {os.path.split(path)[-1]}/{file}:")
    print(f"{[chan for chan in tiff['channels']]}")
    print(f"{tiff['date_time']}")
    print()
    tiff["tiff"].close()
