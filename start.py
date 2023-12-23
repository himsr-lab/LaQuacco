import os
import tifffile
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from typing import Any, Dict

FILES = [
    # r"C:\Users\Christian Rickert\Desktop\Polaris\100923 P9huP54-2 #04 B1A1P1_[8520,46568]_component_data.tif",
    r"/Users/christianrickert/Desktop/Polaris/100923 P9huP54-2 #04 B1A1P1_[8520,46568]_component_data.tif",
    # r"C:\Users\Christian Rickert\Desktop\MIBI\[FOV2-1] UCD134_bottom_sample_SA21-06560_2023-05-24_10.tif",
    # r"/Users/christianrickert/Desktop/MIBI/[FOV2-1] UCD134_bottom_sample_SA21-06560_2023-05-24_10.tif",
]


def get_signal_threshold(array):
    """Return the threshold value for separating background values from signa valuesl.
    In short, we're transforming the positive array values to be normally distributed.
    From this distribution picture a boxplot, where we're treating the background values
    as low-value "outliers" from our high-value signals.
    Keyword arguments:
    array  -- a Numpy array to be normalized
    """
    array_norm, lmbda = sp.stats.boxcox(array[array > 0])  # positive values only
    quartile_one = np.percentile(array_norm, 25)  # Q1
    interquartile_range = sp.stats.iqr(array_norm)  # IQR
    bottom_whisker = quartile_one - 1.5 * interquartile_range
    signal_threshold = sp.special.inv_boxcox(bottom_whisker, lmbda)
    return (
        signal_threshold if not np.isnan(signal_threshold) else np.min(array[array > 0])
    )


for file in sorted(FILES):
    name = os.path.basename(file)
    print(f"\n\tFILE: {name}", flush=True)

    # open TIFF file to extract image information
    with tifffile.TiffFile(file) as tif:
        # use data from first series (of pages) only
        series = tif.series
        pages = series[0].shape[0]
        # access pages of first series
        for chan, page in enumerate(tif.pages[0:pages]):
            # get page name
            page_name = None
            try:
                page_name = page.tags["PageName"].value  # regular TIFF
            except KeyError:
                image_description = page.tags["ImageDescription"].value  # OME-TIFF
                image_dictionary = xmltodict.parse(image_description)
                vendor_id = next(iter(image_dictionary))  # first and only key
                page_name = image_dictionary[vendor_id]["Name"]
            finally:
                if not page_name:
                    page_name = str(chan)

            # get pixel data as flattend Numpy array
            pixels = page.asarray().flatten()
            print(f"{page_name}:\t {get_signal_threshold(pixels)}")
            # pixels = np.log(pixels[pixels > 0])
            plt.hist(pixels, bins=int(10 * np.max(pixels)), color="black")
            plt.axvline(x=get_signal_threshold(pixels), color="red", linestyle="--")
            plt.axvline(
                x=np.percentile(pixels[pixels > 0], 10), color="green", linestyle="--"
            )
            plt.show()
            # print(np.mean(sorted_pixels[: get_bottom_index(sorted_pixels, 10)]))
            # print(np.mean(sorted_pixels[get_top_index(sorted_pixels, 20) :]))
