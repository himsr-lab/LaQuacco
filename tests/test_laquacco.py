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
Summary:    Laboratory Quality Control v2.0 (2024-10-17)
DOI:        # TODO
URL:        https://github.com/himsr-lab/LaQuacco
"""

import os
import shutil
import sys
import time
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
                "mean": 127.630746875,
                "min": 0.0,
            },
            "Channel 2": {
                "max": 255.0,
                "mean": 127.389440625,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_means=img_chans_stats_expected
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 31.580078271995184,
                "band_1": 95.56676157650232,
                "band_2": 159.4273558578761,
                "band_3": 223.61814747166412,
            },
            "Channel 2": {
                "band_0": 31.487712218028722,
                "band_1": 95.53149690102047,
                "band_2": 159.4692337533945,
                "band_3": 223.45975257370876,
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
                "mean": 159.49946312188382,
                "min": 64.0,
            },
            "Channel 2": {
                "max": 192.0,
                "mean": 95.9103757753552,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_limits=chans_limits, chans_means=img_chans_stats_expected
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 87.53852950983661,
                "band_1": 135.54264168416876,
                "band_2": 183.52176530595182,
                "band_3": 231.60101279317698,
            },
            "Channel 2": {
                "band_0": 23.50682044473946,
                "band_1": 71.49815547431852,
                "band_2": 119.5339726620382,
                "band_3": 168.0184582153786,
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
                "mean": 128.0307885584147,
                "min": 64.0,
            },
            "Channel 2": {
                "max": 192.0,
                "mean": 95.9103757753552,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        img_chans_stats_results = laq.get_img_chans_stats(
            image, chans_limits=chans_limits, chans_means=img_chans_stats_expected
        )
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 79.50962669938158,
                "band_1": 111.54628130061045,
                "band_2": 143.51385827161718,
                "band_3": 176.02873325716507,
            },
            "Channel 2": {
                "band_0": 23.50682044473946,
                "band_1": 71.49815547431852,
                "band_2": 119.5339726620382,
                "band_3": 168.0184582153786,
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
            "band_0": 31.382288195187574,
            "band_1": 95.35042657055523,
            "band_2": 159.48075548594107,
            "band_3": 223.58119422209955,
        }
        assert chan_stats_results == chan_stats_expected
