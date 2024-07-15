import tifffile
import xmltodict
import json
import os
import fnmatch


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
    channels = get_chans(tiff)
    return {
        "tiff": tiff,
        "image": image,
        "shape": shape,
        "pages": pages,
        "channels": channels,
    }


def get_chans(tiff):
    """Get the channel names from a TIFF object.

    Keyword arguments:
    tiff -- the TIFF object
    """
    chans = []
    pages = tiff.series[0].pages  # pages of first series
    # OME TIFF
    ome_metadata = tiff.ome_metadata
    if ome_metadata:  # TIFF baseline tag 'ImageDescription'
        ome_dict = xmltodict.parse(ome_metadata)
        try:  # access first image entry
            channel = ome_dict["OME"]["Image"][0]["Pixels"]["Channel"]
        except KeyError:  # fallback to only image entry
            channel = ome_dict["OME"]["Image"]["Pixels"]["Channel"]
        finally:
            chans = [chan["@Name"] for chan in channel]
    # regular TIFF
    elif pages[0].tags.get("PageName", None):  # TIFF extension tag 'PageName'
        for page in pages:
            chans.append(page.aspage().tags["PageName"].value)
    # custom TIFF
    else:
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
FILES.append(get_files(path="./tests", pat="*tif", anti=""))
FILES.append(get_files(path="./tests", pat="*tiff", anti=""))
FILES = [file for sublist in FILES for file in sublist]

for index, FILE in enumerate(FILES):
    print("\n" + str(index) + " " + os.path.basename(FILE))
    tiff = get_tiff(FILE)
    tiff["tiff"].close()
