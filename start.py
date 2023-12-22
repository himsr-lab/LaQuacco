import os
import tifffile
import xmltodict
from typing import Any, Dict

FILES = [
    r"C:\Users\Christian Rickert\Desktop\Polaris\100923 P9huP54-2 #04 B1A1P1_[8520,46568]_component_data.tif",
    r"C:\Users\Christian Rickert\Desktop\MIBI\[FOV2-1] UCD134_bottom_sample_SA21-06560_2023-05-24_10.tif",
]

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
            # get pixel data
            pixels = page.asarray()
