# PPM - Activities Location Significance

## Files

### 01_read_prefix_sublog.py
The main Python script for generating and managing sublogs based on event prefixes. Used to extract and analyse sublogs from larger datasets.

### datasets.json
A JSON file containing detailed information about the datasets, including structure and metadata.

### requirements.txt
Lists the dependencies required to run the scripts and components of the project. Can be used to install packages with `pip install -r requirements.txt`.

## Directories

### config
Contains configuration files to customise parameters and application settings.

### datasets
Stores the original datasets used for analysis and transformation (see `datasets.json`).

### datasets_encoded
A directory where prefixes datasets are saved in encoded formats for ML processing.

### datasets_prefix
Contains datasets generated with event prefixes for ML and sublogs.

### datasets_stats
Includes computed statistics on the original datasets.

### datasets_sublog
Contains the sublogs extracted from the prefixes datasets.

### ml_results
A directory containing the results of ML models applied to encoded prefixes.

### utilities
A collection of scripts and tools for common operations such as preprocessing, conversions, and exploratory analysis.
