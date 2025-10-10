# Laboratory Quality Control
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17298006.svg)](https://doi.org/10.5281/zenodo.17298006)

![image](https://github.com/christianrickert/LaQuacco/assets/19319377/80dad1f9-3ecf-4be1-afbb-46438bd0066f)

## Purpose
LaQuacco produces pixel-based [Levey–Jennings charts](https://en.wikipedia.org/wiki/Laboratory_quality_control#Levey%E2%80%93Jennings_chart) of a dataset of images for quality control and documentation.

## Motivation
Maintaining consistency in immunofluorescence staining and imaging when processing large batches of tissue sections remains a technical challenge. It is therefore important to track the experimental output for any deviations from an optimal outcome. However, manual screening of hundreds of images across dozens of signal channels is not feasible: LaQuacco therefore collects basic statistical parameters from any given image dataset and displays simple diagnostic charts to identify signal outliers (stochastic events) or signal drift (systematic events).

## Installation
Use [`venv`](https://docs.python.org/3/library/venv.html) to create a virtual Python environment for the installation of LaQuacco and its dependencies. The installation is operating system-dependent, but a general workflow follows these steps in macOS:
```zsh
# change to your home directory
cd ~

# download LaQuacco from GitHub
git clone https://github.com/himsr-lab/LaQuacco.git

# check Python version (>=3.12)
python --version

# create virtual environment
python -m venv LaQuacco

# activate virtual environment
. LaQuacco/bin/activate

# install required modules in environment
(LaQuacco) pip install -r LaQuacco/requirements.txt

# change to your LaQuacco directory
(LaQuacco) cd LaQuacco

# test successful installation (all tests must pass)
(LaQuacco) pytest -vv tests

# optional: install libraries for PDF exports using homebrew
(LaQuacco) brew install pandoc
(LaQuacco) brew install mactex
```

## Starting
```zsh
# activate virtual environment
. LaQuacco/bin/activate

# run LaQuacco with Jupyter 
jupyter lab ~/LaQuacco/LaQuacco.ipynb
```

## Workflow
Running LaQuacco consists of two basic steps:
1. Adjust the variables in the `User Input` cell as needed.
2. Select `Run`->`Run All Cells` to execute the notebook.
3. Save your notebook with `File`->`Save Notebook`.
4. Optional: `Save and Export Notebook As`->`PDF`.

## Examples
The first notebook sections provide a high-level overview of your dataset's pixel intensity distributions:

```text
0 => './tests/image_1.ome.tiff'  # 2024-09-22 17:45:28
1 => './tests/image_2.ome.tiff'  # 2024-09-22 17:46:58
2 => './tests/image_3.ome.tiff'  # 2024-09-22 17:48:28
3 => './tests/image_4.ome.tiff'  # 2024-09-22 17:49:58
4 => './tests/image_5.ome.tiff'  # 2024-09-22 17:51:28

Channel 1 @ 0.001s
	max	    255.0 (mean)	    0.0 (std)
	mean	  128.9 (mean)	   18.3 (std)
	min	      0.0 (mean)	    0.0 (std)
Channel 2 @ 0.001s
	max	    255.0 (mean)	    0.0 (std)
	mean	  127.4 (mean)	    0.0 (std)
	min	      0.0 (mean)	    0.0 (std)
```
<img width="1478" height="839" alt="Violin chart for all channels" src="https://github.com/user-attachments/assets/0133e445-f15a-47ec-be53-0367ca21c577" />
<br /><br />

The next notebook section shows classical Levey-Jennings charts for each of your image channels and each of your images. Keep in mind that these charts plot mean pixel values for all pixels, even if large parts of the images contain background signal. In that case, you can increase the channels' `lower` limits to effectively ignore regions with very low signal intensity. Notice that `Channel 1` exhibits two images where the mean signal intensity deviates from the expected mean value:

```text
Channel 1 @ 0.001s
	▲ +1std
		  159.6 -> './tests/image_4.ome.tiff'
	▼ -1std
		  101.9 -> './tests/image_2.ome.tiff'
```
<img width="1478" height="839" alt="Levey-Jennings chart for Channel 1" src="https://github.com/user-attachments/assets/400465b5-f261-47f7-8d7e-58893cd0d90c" />
<br /><br />

The last notebook section contains custom Levey-Jennings charts that evenly break up the summarized data into four distinct channel bands (C-bands), corresponding to mean background (`band_0`), mean low signal (`band_1`), mean medium signal (`band_2`), and mean high signal (`band_3`) intensities. You can now see that for `Channel 1`, `image_2.ome.tiff` deviates from the background and low signal bands, while `image_4` deviates from the medium and high signal bands:

```text
Channel 1 @ 0.001s
	band_3
		▲ +1std
			  226.3 -> './tests/image_4.ome.tiff'
	band_2
		▲ +1std
			  159.7 -> './tests/image_4.ome.tiff'
	band_1
		▼ -1std
			   86.8 -> './tests/image_2.ome.tiff'
	band_0
		▲ +1std
			   41.0 -> './tests/image_2.ome.tiff'
```
<img width="1478" height="839" alt="C-band charts for Channel 1" src="https://github.com/user-attachments/assets/37694701-cd14-456b-869b-2cd2f7c121a6" />
<br /><br />

All charts are interactive while the Jupyter Lab session is running: However, the current Jupyter framework does not allow for active content to be loaded into a new session for security reasons. Instead, a non-interactive version of the last session will be shown.
