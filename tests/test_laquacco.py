import os
import sys
# import warnings

# import laquacco from local path
sys.path.append(os.path.abspath(os.path.join("../", "laquacco")))
import laquacco as laq


class TestLaquacco:
    def test_get_files(self):
        relpath = "./tests"
        files = laq.get_files(path=relpath, pat="*.ome.tiff")
        files_normalized = [file.replace("\\", "/") for file in files]  # Windows
        files_expected = [f"{relpath}/image_{count+1}.ome.tiff" for count in range(5)]
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

        # copies = [copy_file(file) for file in files]
        # images = [get_image(copy) for copy in copies]
        # for image in images:
        #    print(image, "\n")
        #    image["tiff"].close()
        # for copy in copies:
        #    os.remove(copy)
        #    print(f"Removed: {copy}")
