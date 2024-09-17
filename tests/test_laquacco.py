import os
import sys
from datetime import datetime
import warnings

# import laquacco from local path
sys.path.append(os.path.abspath(os.path.join("../", "laquacco")))
import laquacco as laq


class TestLaquacco:
    def test_get_files(self):
        relpath = "./tests"
        files = laq.get_files(path=relpath, pat="*.ome.tiff")
        files_normalized = [file.replace("\\", "/") for file in files]  # Windows
        files_expected = [f"{relpath}/image_{count + 1}.ome.tiff" for count in range(5)]
        assert files_normalized == files_expected

    def test_copy_files(self):
        relpath = "./tests"
        files = laq.get_files(path=relpath, pat="*.ome.tiff")
        copies = [laq.copy_file(file) for file in files]
        for index, copy in enumerate(copies):
            # warnings.warn(f"Copy: {copy}")
            assert os.path.exists(copy)
            assert os.path.getsize(files[index]) == os.path.getsize(copy)
            os.remove(copy)

    def test_get_tiff(self):
        relpath = "./tests"
        files = sorted(laq.get_files(path=relpath, pat="*.ome.tiff"))
        images = [laq.get_image(file) for file in files]
        channels = 3
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
            image["tiff"].close()
