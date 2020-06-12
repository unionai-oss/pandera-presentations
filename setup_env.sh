#!/bin/sh

conda create -n pandera-presentations -c conda-forge \
    python=3.7 \
    ipykernel \
    jupyter \
    jupyterlab \
    jupytext \
    matplotlib \
    pandas \
    pandas-profiling \
    pandera==0.4.2 \
    pyjanitor \
    requests \
    rise \
    seaborn \
    scikit-learn==0.23.0 \
    shap \
    statsmodels

python -m ipykernel install \
    --name 'pandera-presentations' \
    --display-name 'pandera-presentations'

source activate pandera-presentations
