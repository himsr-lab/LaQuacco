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
