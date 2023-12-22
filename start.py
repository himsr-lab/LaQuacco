import os
import tifffile
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict

FILES = [
    r"C:\Users\Christian Rickert\Desktop\Polaris\100923 P9huP54-2 #04 B1A1P1_[8520,46568]_component_data.tif",
    # r"C:\Users\Christian Rickert\Desktop\MIBI\[FOV2-1] UCD134_bottom_sample_SA21-06560_2023-05-24_10.tif",
]


def get_bottom_index(sorted_list, percentage):
    """Return a stop index for a slice of a sorted array with the bottom (percentage) elements.
    Keyword arguments:
    sorted_array  -- a sorted list or Numpy array
    percentage  -- the upper percentage limit for the slice
    """
    len_sorted_list = len(sorted_list)
    return int(percentage / 100 * len_sorted_list)


def get_top_index(sorted_list, percentage):
    """Return a start index for a slice of a sorted array with the top (percentage) elements.
    Keyword arguments:
    sorted_array  -- a sorted list or Numpy array
    percentage  -- the lower percentage limit for the slice
    """
    len_sorted_list = len(sorted_list)
    percentage = percentage or len_sorted_list
    return int((100 - percentage) / 100 * len_sorted_list)


for file in sorted(FILES):
    name = os.path.basename(file)
    print(f"\n\tFILE: {name}", flush=True)

    # open TIFF file to extract image information
    with tifffile.TiffFile(file) as tif:
        # use data from first series (of pages) only
        series = tif.series
        pages = series[0].shape[0]
        # access pages of first series
        for page in tif.pages[0:pages]:
            # get page name
            try:
                page_name = page.tags["PageName"].value  # regular TIFF
            except KeyError:
                image_description = page.tags["ImageDescription"].value  # OME-TIFF
                image_dictionary = xmltodict.parse(image_description)
                vendor_id = next(iter(image_dictionary))  # first and only key
                page_name = image_dictionary[vendor_id]["Name"]
            print(page_name)
            # get pixel data as flattend Numpy array
            pixels = page.asarray().flatten()
            # pixels = np.log(pixels[pixels > 0])
            plt.hist(pixels, bins=256)
            plt.show()
            # sorted_pixels = np.sort(pixels)
            # print(np.mean(sorted_pixels[: get_bottom_index(sorted_pixels, 10)]))
            # print(np.mean(sorted_pixels[get_top_index(sorted_pixels, 20) :]))
            break
