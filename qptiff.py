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
Summary:    Laboratory Quality Control v2.0 (2024-11-05)
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

    * Annotations @ROIAnnotation
      * Fields @RectangleAnnotation
        * Histories @FlaggedForAnalysis
        * Bounds (coordinates)
          * ...

    Keyword arguments:
    field -- single `RectangleAnnotation` data
    """
    return field["Bounds"]


def get_coordinates(bounds, resolution, offsets):
    """Get the top-left and bottom-right coordinates (in pixels) of
    the current rectangular's bounds as a tuple of tuples.

    * Annotations @ROIAnnotation
      * Fields @RectangleAnnotation
        * Histories @FlaggedForAnalysis
        * Bounds (Rectangle)
          * Origin
            * X, Y
          * Size
            * Width, Height

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
    """Get the list of fields from a list of annotations

    * Annotations @ROIAnnotation
      * Fields @RectangleAnnotation
        * Histories @FlaggedForAnalysis
        * ...

    Keyword arguments:
    annos -- list of `ROIAnnotation` data
    """
    fields = [
        field_i
        for anno in annos
        if anno.get("Fields") is not None  # missing if deleted
        and anno.get("@subtype") == "ROIAnnotation"
        for field_i in (
            [anno["Fields"]["Fields-i"]]
            if isinstance(anno["Fields"]["Fields-i"], dict)  # list only
            else anno["Fields"]["Fields-i"]
        )
        if field_i.get("@subtype") == "RectangleAnnotation"
        and get_histories(field_i)[-1].get("Type") == "FlaggedForAnalysis"  # current
    ]
    return fields


def get_histories(field):
    """Get the list histories from field data.
    The history section contains information about the author and
    the last type of annotation for the current field element.

    * Annotations @ROIAnnotation
      * Fields @RectangleAnnotation
        * Histories @FlaggedForAnalysis (last)
          * ...

    Keyword arguments:
    field -- single `RectangleAnnotation` data
    """
    histos_i = field["History"]["History-i"]
    histos_i = [histos_i] if isinstance(histos_i, dict) else histos_i  # list only
    histos = [histo_i for histo_i in histos_i]
    return histos


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


def get_rectangles(file, offsets):
    """Get a list of rectangular bounds annotations. We are only
    considering the rectangles of type `FlaggedForAnalysis`.

    * Annotations @ROIAnnotation
      * Fields @RectangleAnnotation
        * Histories @FlaggedForAnalysis
          * Bounds (coordinates)

    Keyword arguments:
    file -- annotation XML file
    offsets -- tuple of X and Y (stage) offsets in pixels
    """
    annos = read_annotations(file)
    rectangles = [
        get_coordinates(get_bounds(field), float(field["Resolution"]), offsets)
        for field in get_fields(annos)
    ]
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

    * Annotations @ROIAnnotation
      * ...

    Keywords:
    file -- annotation XML file path

    """
    annos = []
    with open(file, "r", encoding="utf-8") as xml:
        xml_annos = xmltodict.parse(xml.read())
        annos_i = xml_annos["AnnotationList"]["Annotations"]["Annotations-i"]
        annos_i = [annos_i] if isinstance(annos_i, dict) else annos_i  # list only
        annos = [
            anno_i for anno_i in annos_i if anno_i.get("@subtype") == "ROIAnnotation"
        ]
    return annos
