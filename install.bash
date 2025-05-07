#!/bin/bash
pip install -r requirements.txt --constraint constraints.txt

cd ~
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .


#!/bin/bash
set -e

# 0) (optional) allow the broken 'sklearn' stub if needed
# export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# 1) Pin and install Torch 1.7.1 (and matching torchvision & torchaudio)
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 --no-deps

# 2) Install everything else but never touch torch/torchvision/torchaudio
pip install -r requirements.txt --no-deps

# 3) Clone & editable-install your repo also without pulling in Torch again
cd ~
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e . --constraint ../constraints.txt
