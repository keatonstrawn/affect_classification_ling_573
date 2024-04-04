#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Affect
conda env update --file requirements.yml --prune
