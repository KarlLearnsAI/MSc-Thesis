# Installation Guide for Extended VLMaps (including OpenSeeD integration)

This guide details how to set up the extended VLMaps codebase and run the entire pipeline. The instructions have been tested on Ubuntu 20.04 and Windows 11.

## 1. Prerequisites
- **Operating system:** Linux or Windows
- **Python:** 3.8.20
- **GPU:** CUDA-compatible GPU (tested with CUDA 11.3) and drivers installed
- **Tools:** Git and Conda (from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda)

## 2. Clone the repository
```bash
git clone https://github.com/KarlLearnsAI/MSc-Thesis
cd MSc-Thesis
```

## 3. Recreate the conda environment
An `environment.yml` file is provided to reproduce the main dependencies and automatically download all required sub‑dependencies.
```bash
conda env create -f environment.yml
conda activate vlmaps6
```

## 5. Run the entire extended VLMaps codebase
Make sure to change flags, depending on if you want to run all individual VLM evaluations and if it is the first time running the notebook. In will download all the neccessary data on the first run.
Execute all cells inside the modular-extended-VLMaps.ipynb notebook.