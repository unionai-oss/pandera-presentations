#!/bin/bash

notebook_path=$1
jupyter nbconvert $notebook_path \
    --to slides \
    --SlidesExporter.reveal_theme=simple \
    --SlidesExporter.reveal_transition=none \
    --SlidesExporter.reveal_scroll=True \
    --output-dir slides \
    --TagRemovePreprocessor.remove_input_tags={\"hide_input\"}
