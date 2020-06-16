#!/bin/sh
# sets up local conda environment

conda create -n pandera-presentations -c conda-forge \
    python=3.7 \
    ipykernel \
    jupyter \
    jupyterlab \
    jupytext \
    matplotlib \
    nodejs

python -m ipykernel install \
    --name 'pandera-presentations' \
    --display-name 'pandera-presentations'

source activate pandera-presentations
