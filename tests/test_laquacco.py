import os
import sys
from datetime import datetime
import warnings
import numpy as np

# import laquacco from local path
sys.path.append(os.path.abspath(os.path.join("../", "laquacco")))
import laquacco as laq


class TestLaquacco:
    files = []
    images = []
    imgs_chans_stats = []

    def test_get_files(self):
        relpath = "./tests"
        self.files = laq.get_files(path=relpath, pat="*.ome.tiff")
        files_normalized = [file.replace("\\", "/") for file in self.files]  # Windows
        files_expected = [f"{relpath}/image_{count + 1}.ome.tiff" for count in range(5)]
        assert files_normalized == files_expected

    def test_copy_files(self):
        copies = [laq.copy_file(file) for file in self.files]
        for index, copy in enumerate(copies):
            assert os.path.exists(copy)
            assert os.path.getsize(self.files[index]) == os.path.getsize(copy)
            os.remove(copy)

    def test_get_tiff(self):
        channels = 3
        files = sorted(self.files)  # order matters for `datetimes`
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
            assert image["image"] == files[index]
            assert image["tiff"].is_ome
            # image["tiff"].close()

    def test_get_chan_stats(self):
        np.random.seed(42)  # set fixed seed
        test_array = np.random.randint(0, 256, size=512 * 512)
        # raw stats (based on channel values)
        chan_stats = laq.get_chan_stats(test_array)
        chan_stats_expected = {"max": 255, "mean": 127.86960567684419, "min": 1}
        assert chan_stats == chan_stats_expected
        # group stats (based on channel averages)
        chan_stats = laq.get_chan_stats(
            test_array, {"max": 192, "mean": 128, "min": 64}
        )
        chan_stats_expected = {
            "band_0": 64.45206899442832,
            "band_1": 149.47300991567047,
            "band_2": 191.9446401610468,
            "band_3": 234.40679147117572,
        }
        assert chan_stats == chan_stats_expected
