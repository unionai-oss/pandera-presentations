#!/bin/sh

conda create -n pandera-presentations -c conda-forge \
    python=3.7 \
    jupyter \
    jupyterlab \
    pandas \
    pandera==0.3.0 \
    scikit-learn
