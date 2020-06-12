#!/bin/bash

notebook_path=$1
jupyter nbconvert $notebook_path \
    --to slides --SlidesExporter.reveal_theme=simple \
    --output-dir slides \
    --TagRemovePreprocessor.remove_input_tags={\"hide_input\"}
