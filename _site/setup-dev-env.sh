#!/bin/sh
# sets up local conda environment

conda create -n pandera-presentations -c conda-forge \
    python=3.9 \
    ipykernel \
    jupyter \
    jupyterlab \
    jupytext \
    matplotlib \
    nodejs

source activate pandera-presentations
conda env update -n pandera-presentations -f environment.yml
# pip install here due to conflict with pandera
pip install shap

python -m ipykernel install \
    --name 'pandera-presentations' \
    --display-name 'pandera-presentations'
