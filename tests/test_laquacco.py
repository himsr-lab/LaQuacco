import os
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
            os.remove(copy)

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

    def test_get_imgs_chan_stats(self):
        imgs_chans_stats = {
            "image_1": {"channel_1": {"stat_1": 1.0, "stat_2": 3.0}},
            "image_2": {"channel_1": {"stat_1": 2.0}},
        }
        imgs_chan_stats_result = laq.get_imgs_chan_stats(
            imgs_chans_stats, "channel_1", "stat_1"
        )
        imgs_chan_stats_expected = np.array([1.0, 2.0])
        assert np.array_equal(imgs_chan_stats_result, imgs_chan_stats_expected)
        imgs_chan_stats_result = laq.get_imgs_chan_stats(
            imgs_chans_stats, "channel_1", "stat_2"
        )
        imgs_chan_stats_expected = np.array([3.0, np.nan])
        assert np.array_equal(
            imgs_chan_stats_result, imgs_chan_stats_expected, equal_nan=True
        )

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
        img_chans_stats_bands = laq.get_img_chans_stats(
            image, chans_stats=img_chans_stats_expected
        )
        for chan in image["channels"]:
            img_chans_stats_results[chan].update(img_chans_stats_bands[chan])
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": 31.580078271995184,
                "band_1": 95.56676157650232,
                "band_2": 159.4273558578761,
                "band_3": 223.61814747166412,
                "max": 255.0,
                "mean": 127.630746875,
                "min": 0.0,
            },
            "Channel 2": {
                "band_0": 31.487712218028722,
                "band_1": 95.53149690102047,
                "band_2": 159.4692337533945,
                "band_3": 223.45975257370876,
                "max": 255.0,
                "mean": 127.389440625,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        # limited interval
        chans_limits = {
            "Channel 1": {"lower": 64, "upper": None},
            "Channel 2": {"lower": None, "upper": 192},
            "Channel 3": {"lower": 64, "upper": 192},
        }
        img_chans_stats_results = laq.get_img_chans_stats(image, chans_limits)
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
        img_chans_stats_bands = laq.get_img_chans_stats(
            image, chans_limits, img_chans_stats_expected
        )
        for chan in image["channels"]:
            img_chans_stats_results[chan].update(img_chans_stats_bands[chan])
        img_chans_stats_expected = {
            "Channel 1": {
                "band_0": None,
                "band_1": 79.50962669938158,
                "band_2": 119.57328957836118,
                "band_3": 199.52674623752688,
                "max": 255.0,
                "mean": 159.49946312188382,
                "min": 64.0,
            },
            "Channel 2": {
                "band_0": 24.014317055334363,
                "band_1": 72.51777903909719,
                "band_2": 120.52741301159845,
                "band_3": 168.50567650594363,
                "max": 192.0,
                "mean": 95.9103757753552,
                "min": 0.0,
            },
        }
        assert img_chans_stats_results == img_chans_stats_expected
        image["tiff"].close()

    def test_get_time_left(self):
        start = time.time() - 86_523  # begin of tracking
        get_time_left_result = laq.get_time_left(start, current=1, total=2)
        get_time_left_expected = "1d 2m 3s"
        assert get_time_left_result == get_time_left_expected

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
        chan_stats_results = laq.get_chan_stats(test_frame, {"max": 255, "min": 0})
        chan_stats_expected = {
            "band_0": 31.382288195187574,
            "band_1": 95.35042657055523,
            "band_2": 159.48075548594107,
            "band_3": 223.58119422209955,
        }
        assert chan_stats_results == chan_stats_expected
