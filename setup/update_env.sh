#!/bin/sh
# Example: source ~/miniconda3/etc/profile.d/conda.sh
source $1
conda activate Affect
conda env update --file requirements.yml --prune
