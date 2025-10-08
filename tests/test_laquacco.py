"""
Copyright 2023 The Regents of the University of Colorado

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
Summary:    Laboratory Quality Control v2.1 (2025-10-08)
DOI:        # TODO
URL:        https://github.com/himsr-lab/LaQuacco
"""

import nbformat
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import polars as pl

# import laquacco from local path
sys.path.append(os.path.abspath(os.path.join("../", "laquacco")))
import laquacco as laq


class TestLaquacco:
    def test_copy_files(self):
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        copies = [laq.copy_file(file) for file in files]
        for index, copy in enumerate(copies):
            assert os.path.exists(copy)
            assert os.path.getsize(files[index]) == os.path.getsize(copy)
            dirname = os.path.dirname(copy)
            shutil.rmtree(dirname)
            assert not os.path.exists(dirname)

    def test_get_expo_si(self):
        expo_si_result = laq.get_expo_si()
        expo_si_expected = (1.0, "s")
        assert expo_si_result == expo_si_expected
        expo_si_result = laq.get_expo_si(expo=(1_000_000, "µs"))
        assert expo_si_result == expo_si_expected

    def test_get_files(self):
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        files_normalized = [file.replace("\\", "/") for file in files]  # Windows
        files_expected = [
            f"{relpath}/image_{count + 1}.ome.tiff" for count in range(len(files))
        ]
        assert files_normalized == files_expected

    def test_get_img(self):
        channels = 2
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        images = [laq.get_img(file) for file in files]
        channels_expected = [f"Channel {count + 1}" for count in range(channels)]
        exposures_expected = [(0.001, "s") for count in range(channels)]
        datetimes_expected = [
            [datetime(2024, 9, 22, 17, 45, 28) for _ in range(channels)],
            [datetime(2024, 9, 22, 17, 46, 58) for _ in range(channels)],
            [datetime(2024, 9, 22, 17, 48, 28) for _ in range(channels)],
            [datetime(2024, 9, 22, 17, 49, 58) for _ in range(channels)],
            [datetime(2024, 9, 22, 17, 51, 28) for _ in range(channels)],
        ]
        for index, image in enumerate(images):
            assert image["channels"] == channels_expected
            assert image["exposures"] == exposures_expected
            assert image["datetimes"] == datetimes_expected[index]
            assert image["file"] == files[index]
            assert image["tiff"].is_ome
            image["tiff"].close()

    def test_get_img_chans_stats(self):
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        image = laq.get_img(files[0])
        # unlimited
        img_chans_stats_results = laq.get_img_chans_stats(image)
        img_chans_stats_expected = {
            "Channel 1": {
                "max": 255.0,
                "mean": 127.63074493408203,
                "min": 0.0,
            },
            "Channel 2": {
                "max": 255.0,
                "mean": 127.38944244384766,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_means=img_chans_stats_expected
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 31.580078125,
                "band_1": 95.56676483154297,
                "band_2": 159.42735290527344,
                "band_3": 223.61814880371094,
                "lim_0": 63.75,
                "lim_1": 127.5,
                "lim_2": 191.25,
            },
            "Channel 2": {
                "band_0": 31.487712860107422,
                "band_1": 95.531494140625,
                "band_2": 159.46923828125,
                "band_3": 223.45974731445312,
                "lim_0": 63.75,
                "lim_1": 127.5,
                "lim_2": 191.25,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        # unlimited with masking
        annos = np.array([((0, 0), (100, 100)), ((300, 700), (400, 800))])
        img_chans_stats_results = laq.get_img_chans_stats(image, annos=annos)
        img_chans_stats_expected = {
            "Channel 1": {
                "max": 255.0,
                "mean": 127.47418212890625,
                "min": 0.0,
            },
            "Channel 2": {
                "max": 255.0,
                "mean": 127.07420349121094,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        img_chans_stats_results = laq.get_img_chans_stats(
            image, annos=annos, chans_means=img_chans_stats_expected
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 31.92799186706543,
                "band_1": 95.57157135009766,
                "band_2": 159.35067749023438,
                "band_3": 223.51089477539062,
                "lim_0": 63.75,
                "lim_1": 127.5,
                "lim_2": 191.25,
            },
            "Channel 2": {
                "band_0": 31.44708251953125,
                "band_1": 95.59882354736328,
                "band_2": 159.9837188720703,
                "band_3": 223.3043670654297,
                "lim_0": 63.75,
                "lim_1": 127.5,
                "lim_2": 191.25,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        # specific limits
        chans_limits = {
            "Channel 1": {"lower": 64, "upper": None},
            "Channel 2": {"lower": None, "upper": 192},
            "Channel 3": {"lower": 64, "upper": 192},
        }
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_limits=chans_limits
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "max": 255.0,
                "mean": 159.4994659423828,
                "min": 64.0,
            },
            "Channel 2": {
                "max": 192.0,
                "mean": 95.9103775024414,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_limits=chans_limits, chans_means=img_chans_stats_expected
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 87.53852844238281,
                "band_1": 135.5426483154297,
                "band_2": 183.52175903320312,
                "band_3": 231.60101318359375,
                "lim_0": 111.75,
                "lim_1": 159.5,
                "lim_2": 207.25,
            },
            "Channel 2": {
                "band_0": 23.506820678710938,
                "band_1": 71.49815368652344,
                "band_2": 119.53397369384766,
                "band_3": 168.01846313476562,
                "lim_0": 48.0,
                "lim_1": 96.0,
                "lim_2": 144.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        # general limits
        chans_limits = {
            "*": {"lower": 64, "upper": 192},
            "Channel 2": {"lower": None, "upper": 192},
        }
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_limits=chans_limits
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "max": 192.0,
                "mean": 128.03079223632812,
                "min": 64.0,
            },
            "Channel 2": {
                "max": 192.0,
                "mean": 95.9103775024414,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_limits=chans_limits, chans_means=img_chans_stats_expected
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 79.50962829589844,
                "band_1": 111.54627990722656,
                "band_2": 143.51385498046875,
                "band_3": 176.0287322998047,
                "lim_0": 96.0,
                "lim_1": 128.0,
                "lim_2": 160.0,
            },
            "Channel 2": {
                "band_0": 23.506820678710938,
                "band_1": 71.49815368652344,
                "band_2": 119.53397369384766,
                "band_3": 168.01846313476562,
                "lim_0": 48.0,
                "lim_1": 96.0,
                "lim_2": 144.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        image["tiff"].close()

    def test_get_time_left(self):
        start = time.time() - 86_523  # begin of tracking
        get_time_left_result = laq.get_time_left(start, current=1, total=2)
        get_time_left_expected = "1d 2m 3s"
        assert get_time_left_result == get_time_left_expected

    def test_set_chan_interval(self):
        np.random.seed(42)
        test_array = np.random.randint(0, 256, size=512 * 512)
        test_frame = pl.from_numpy(
            test_array.ravel(), schema=["pixls"], orient="col"
        ).lazy()
        row = pl.col("pixls")
        query = [
            row.max().alias("max"),
            row.mean().alias("mean"),
            row.min().alias("min"),
        ]
        # unlimited with None
        limits = None  # {}
        test_filter = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(test_filter, query)
        stats_expected = {"max": 255, "mean": 127.37548065185547, "min": 0}
        assert stats_results == stats_expected
        # unlimited with dictionary
        limits = {"lower": None, "upper": None}
        test_filter = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(test_filter, query)
        stats_expected = {"max": 255, "mean": 127.37548065185547, "min": 0}
        assert stats_results == stats_expected
        limits = {"upper": 192}
        test_filter = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(test_filter, query)
        stats_expected = {"max": 192, "mean": 95.99974737395223, "min": 0}
        assert stats_results == stats_expected
        # lower and upper limit
        limits = {"lower": 64, "upper": 192}
        test_filter = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(test_filter, query)
        stats_expected = {"max": 192, "mean": 127.89909212343498, "min": 64}
        assert stats_results == stats_expected

    def test_get_chan_stats(self):
        np.random.seed(42)
        test_array = np.random.randint(0, 256, size=512 * 512)
        # raw stats (based on channel values)
        chan_stats_results = laq.get_chan_stats(test_array)
        chan_stats_expected = {"max": 255, "mean": 127.37548065185547, "min": 0}
        assert chan_stats_results == chan_stats_expected
        # group stats (based on channel averages)
        chan_stats_results = laq.get_chan_stats(
            test_array, chan_means={"max": 255, "mean": 127.37548065185547, "min": 0}
        )
        chan_stats_expected = {
            "band_0": 31.382287979125977,
            "band_1": 95.35042572021484,
            "band_2": 159.4807586669922,
            "band_3": 223.58119201660156,
            "lim_0": 63.75,
            "lim_1": 127.5,
            "lim_2": 191.25,
        }
        assert chan_stats_results == chan_stats_expected

    def test_jupyer_output(self):
        notebook_results = os.path.abspath("./Laquacco_results.ipynb")
        notebook_expected = os.path.abspath("./tests/Laquacco_expected.ipynb")

        if os.path.exists(notebook_results) and os.path.exists(notebook_expected):

            def read_notebook(file):
                """Read output data from a Jupyter lab notebook.
                Strip all run-time information and widget output that contains
                inconsistenr hash values, binary output, and version numbers.

                Keyword arguments:
                file -- Jupyter lab notebook file
                """
                with open(file, "r", encoding="utf-8") as nb:
                    nb_data = nbformat.read(nb, as_version=nbformat.NO_CONVERT)
                    # remove non-text output (widgets output is not reproducible)
                    outputs = []
                    for cell in nb_data.cells:
                        if cell.cell_type == "code" and "outputs" in cell:
                            # Collect only text outputs
                            for output in cell["outputs"]:
                                if output.get("output_type") == "stream":
                                    outputs.append(output.get("text", ""))
                                elif output.get(
                                    "output_type"
                                ) == "execute_result" and "text/plain" in output.get(
                                    "data", {}
                                ):
                                    outputs.append(output["data"]["text/plain"])
                    return outputs

            assert read_notebook(notebook_results) == read_notebook(notebook_expected)
        else:
            warnings.warn(
                "Jupyter output not tested. - Run `Laquacco.ipynb` and save as `Laquacco_results.ipynb` "
                "to compare against `./tests/Laquacco_expected.ipynb` with the next test iteration!"
            )
            assert True
