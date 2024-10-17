"""
Copyright 2024 The Regents of the University of Colorado

This file is part of LaQuacco.

LaQuacco is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LaQuacco is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author:     Christian Rickert <christian.rickert@cuanschutz.edu>
Group:      Human Immune Monitoring Shared Resource (HIMSR)
            University of Colorado, Anschutz Medical Campus

Title:      LaQuacco
Summary:    Laboratory Quality Control v2.0 (2024-10-17)
DOI:        # TODO
URL:        https://github.com/himsr-lab/LaQuacco
"""

# modules
import math
import os
import xmltodict
import numpy as np


# functions
def get_bounds(field):
    """Get the top-left and bottom-right coordinates (in microns) of
    the current rectangular field.

    Keyword arguments:
    field -- field data
    """
    return field["Bounds"]


def get_coordinates(bounds, resolution, offsets):
    """Get the top-left and bottom-right coordinates (in pixels) of
    the current rectangular's bounds as a tuple of tuples.

    Keyword arguments:
    bounds -- rectangular's bounds in micrometers
    resolution -- resolution given in micrometers per pixel
    offsets  -- tuple of X and Y (stage) offsets in pixels
    """
    x1 = float(bounds["Origin"]["X"]) / resolution - offsets[0]  # px
    y1 = float(bounds["Origin"]["Y"]) / resolution - offsets[1]  # px
    x2 = x1 + float(bounds["Size"]["Width"]) / resolution  # px
    y2 = y1 + float(bounds["Size"]["Height"]) / resolution  # px
    return ((math.floor(x1), math.floor(y1)), (math.floor(x2), math.floor(y2)))


def get_fields(annos):
    """Get the list of annotation fields from annotation data.

    Keyword arguments:
    annos -- annotation data
    """
    return annos["Fields"]["Fields-i"]


def get_histories(fields):
    """Get the list histories from fields data.
    The history section contains information about the author and
    the type of annotation for the current field element.

    Keyword arguments:
    fields -- fields data
    """
    return fields["History"]["History-i"]


def get_offsets(tags):
    """Get the stage offsets in pixels from the corresponding TIFF tags.

    Keyword arguments:
    tags -- dictionary of TIFF tags
    """
    res_unit = tags["ResolutionUnit"]
    if res_unit == "<RESUNIT.CENTIMETER: 3>":
        to_microns = 10_000  # microns per centimeter
    elif res_unit == "<RESUNIT.INCH: 2>":
        to_microns = 25_400  # microns per inch
    else:
        to_microns = 1  # unknown resolution unit
    xres = (
        float(tags["XResolution"][0]) / float(tags["XResolution"][1]) * to_microns
    )  # px / µm
    yres = (
        float(tags["YResolution"][0]) / float(tags["YResolution"][1]) * to_microns
    )  # px / µm
    xpos = (
        float(tags["XPosition"][0]) / float(tags["XPosition"][1]) * to_microns * xres
    )  # px
    ypos = (
        float(tags["YPosition"][0]) / float(tags["YPosition"][1]) * to_microns * yres
    )  # px
    return (xpos, ypos)


def get_rectangles(annos, offsets, anno_type):
    """Get a list of rectangular bounds annotations. We are only
    considering the rectangles of type `FlaggedForAnalysis`.

    Keyword arguments:
    annos -- annotation data
    offsets  -- tuple of X and Y (stage) offsets in pixels
    anno_type -- string with annotation type selected in Phenochart
    """
    rectangles = []
    fields = get_fields(annos)
    for field in fields:
        histories = get_histories(field)
        for history in histories:
            if "Type" in history and history["Type"] == anno_type:
                resolution = float(annos["Resolution"])  # µm / px
                bounds = get_bounds(field)
                coordinates = get_coordinates(bounds, resolution, offsets)
                rectangles.append(coordinates)
    return np.array(rectangles)


def get_xml(file):
    """Get annotation XML file path from image file path.

    Keywords:
    file -- image file path
    """
    path, img = os.path.split(file)
    name, ext = os.path.splitext(img)
    anno_xml = os.path.join(path, name + "_annotations.xml")
    return anno_xml


def read_annotations(file):
    """Get a list of annotations from a `_annotations.xml` file.

    Keyword arguments:
    file -- annotations file
    """
    with open(file, "r", encoding="utf-8") as annos:
        xml_annos = xmltodict.parse(annos.read())
        annos = xml_annos["AnnotationList"]["Annotations"]["Annotations-i"]
        return annos
