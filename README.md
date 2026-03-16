# gutBrainPipeline
Analysis package for gut-brain processing

# gutBrain_cellSelection_example.ipy

This notebook runs through example use of the pipeline, assuming all relevant data has been placed in an identified folder. The necessary data are:
1. cells0_clean.hdf5
2. volume0.hdf5
3. eP.26chFlt-v10
4. parameters.pickle

The notebook goes through the following steps:
1. Loading data
2. Aligning of ephys data and imaging data
3. Creating of time series by downsampling to the frequency of the imaging data
4. Running the LOOCV parts model
5. Establishing a threshold for significance and extracting relevant cells
6. Plotting results




# eclass: Electrophysiology Data Loader for Larval Zebrafish

This repository contains `eclass.py`, a standalone Python script designed to load, process, and generate metadata for larval zebrafish electrophysiology and stimulus recordings. 

This script is provided as a companion tool for the datasets published on Figshare, supporting the manuscript:
**"Whole-brain, all-optical interrogation of neuronal dynamics underlying gut and vascular interoception in zebrafish"**

## Overview
While this script originated as part of a larger, internal analysis pipeline, `eclass.py` has been isolated here to allow researchers to easily access and interact with the raw `.chFlt` electrophysiology data files hosted on Figshare without needing to install a complex package.

The script performs two main functions:
1. **Metadata Generation:** Reads the raw data file and generates a `channel_meta.npy` dictionary containing sampling rates, fictive swim channels, and stimulus tags (e.g., UV onset, visual grating velocity, camera triggers).
2. **Data Initialization:** Creates an `Ephys` object that structures the data for immediate downstream analysis in Python.

## Requirements
* Python 3.x
* `numpy`

## Quick Start Guide

To reproduce the initial data loading step for the associated Figshare datasets:

1. **Download the script:** Clone this repository or download `eclass.py` directly to your local working directory.
2. **Download the data:** Download the desired electrophysiology data file (ending in `*chFlt*`) from the [Figshare Dataset Repository](#) *(Note: Insert your Figshare DOI link here)*. Place it in a subfolder named `downloaded_data/`.
3. **Run the demo:** Save the following code as `demo_script.py` in the same directory as `eclass.py` and run it.

```python
import os
import numpy as np
from glob import glob
import eclass as ec

# 1. Set paths
DATA_DIRECTORY = './downloaded_data/' 
ephys_path = glob(os.path.join(DATA_DIRECTORY, '*chFlt*'))[0]
dirs = {'ephys': DATA_DIRECTORY}

# 2. Generate metadata and load the Ephys object
ec.create_meta(dirs['ephys'], ephys_path)
fpath = os.path.join(dirs['ephys'], 'channel_meta.npy')
meta = np.load(fpath, allow_pickle=True).item()
E = ec.Ephys(ephys_path=ephys_path, meta=meta, dirs=dirs)

print("Data successfully loaded! The 'E' object is ready for analysis.")
