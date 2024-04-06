#!/bin/sh
# Example: source ~/miniconda3/etc/profile.d/conda.sh
source $1
conda remove -n Affect --all
conda env create -f requirements.yml
conda activate Affect
python -m nltk.downloader all
python -m spacy download en_core_web_sm
