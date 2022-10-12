#!/bin/bash

notebook_path=$1
jupytext --sync $notebook_path
