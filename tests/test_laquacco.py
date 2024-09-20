import os
import sys
from datetime import datetime
import numpy as np
import polars as pl

# import laquacco from local path
sys.path.append(os.path.abspath(os.path.join("../", "laquacco")))
import laquacco as laq


class TestLaquacco:
    # test_array = np.random.randint(0, 256, size=512 * 512)
    # create lazy DataFrame from NumPy array without cloning
    # test_frame = pl.DataFrame(test_array.ravel(), schema=["pixls"], orient="col").lazy()

    def test_get_files(self):
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        files_normalized = [file.replace("\\", "/") for file in files]  # Windows
        files_expected = [f"{relpath}/image_{count + 1}.ome.tiff" for count in range(5)]
        assert files_normalized == files_expected

    def test_copy_files(self):
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        copies = [laq.copy_file(file) for file in files]
        for index, copy in enumerate(copies):
            assert os.path.exists(copy)
            assert os.path.getsize(files[index]) == os.path.getsize(copy)
            os.remove(copy)

    def test_get_image(self):
        channels = 3
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        images = [laq.get_image(file) for file in files]
        channels_expected = [f"Channel {count + 1}" for count in range(channels)]
        exposures_expected = [(0.0005, "s") for count in range(channels)]
        datetimes_expected = [
            [datetime(2024, 9, 17, 14, 43, 14) for _ in range(channels)],
            [datetime(2024, 9, 17, 14, 43, 16) for _ in range(channels)],
            [datetime(2024, 9, 17, 14, 43, 17) for _ in range(channels)],
            [datetime(2024, 9, 17, 14, 43, 19) for _ in range(channels)],
            [datetime(2024, 9, 17, 14, 43, 20) for _ in range(channels)],
        ]
        for index, image in enumerate(images):
            assert image["channels"] == channels_expected
            assert image["exposures"] == exposures_expected
            assert image["datetimes"] == datetimes_expected[index]
            assert image["file"] == files[index]
            assert image["tiff"].is_ome
            # image["tiff"].close()

    def test_query_results(self):
        np.random.seed(42)
        test_array = np.random.randint(0, 256, size=512 * 512)
        test_frame = pl.DataFrame(
            test_array.ravel(), schema=["pixls"], orient="col"
        ).lazy()
        query = [
            pl.col("pixls").mean().alias("mean"),
        ]
        stats_results = laq.get_query_results(test_frame, query)
        expected_results = {"mean": 127.37548065185547}
        assert stats_results == expected_results

    def test_set_chan_interval(self):
        np.random.seed(42)
        test_array = np.random.randint(0, 256, size=512 * 512)
        test_frame = pl.DataFrame(
            test_array.ravel(), schema=["pixls"], orient="col"
        ).lazy()
        query = [
            pl.col("pixls").max().alias("max"),
            pl.col("pixls").mean().alias("mean"),
            pl.col("pixls").min().alias("min"),
        ]
        # unlimited
        limits = {"lower": None, "upper": None}
        pixls = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(pixls, query)
        stats_expected = {"max": 255, "mean": 127.37548065185547, "min": 0}
        assert stats_results == stats_expected
        # lower limit
        limits = {"lower": 64}
        pixls = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(pixls, query)
        stats_expected = {"max": 255, "mean": 159.2938392720988, "min": 64}
        assert stats_results == stats_expected
        # upper limit
        limits = {"upper": 192}
        pixls = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(pixls, query)
        stats_expected = {"max": 192, "mean": 95.99974737395223, "min": 0}
        assert stats_results == stats_expected
        # lower and upper limit
        limits = {"lower": 64, "upper": 192}
        pixls = laq.set_chan_interval(test_frame, limits)
        stats_results = laq.get_query_results(pixls, query)
        stats_expected = {"max": 192, "mean": 127.89909212343498, "min": 64}
        assert stats_results == stats_expected

    def test_get_chan_stats(self):
        np.random.seed(42)
        test_array = np.random.randint(0, 256, size=512 * 512)
        test_frame = pl.DataFrame(
            test_array.ravel(), schema=["pixls"], orient="col"
        ).lazy()
        # raw stats (based on channel values)
        chan_stats = laq.get_chan_stats(test_frame)
        chan_stats_expected = {"max": 255, "mean": 127.37548065185547, "min": 0}
        assert chan_stats == chan_stats_expected
        # group stats (based on channel averages)
        chan_stats = laq.get_chan_stats(test_frame, {"max": 255, "min": 0})
        chan_stats_expected = {
            "band_0": 31.382288195187574,
            "band_1": 95.35042657055523,
            "band_2": 159.48075548594107,
            "band_3": 223.58119422209955,
        }
        assert chan_stats == chan_stats_expected


"""
    def test_get_img_chans_stats(self):
        # img_chans_stats = {}
        # img_chans_stats_expected = {}
        # for image in self.images:
        #    img_chans_stats.update(laq.get_img_chans_stats(image))
        # assert img_chans_stats == img_chans_stats_expected
        assert self.images
"""
