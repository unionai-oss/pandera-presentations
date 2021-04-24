#!/bin/bash

notebook_path=$1
jupyter nbconvert $notebook_path \
    --execute \
    --to slides \
    --SlidesExporter.reveal_theme=simple \
    --SlidesExporter.reveal_transition=none \
    --SlidesExporter.reveal_scroll=True \
    --template-file .jupyter/slide_template.html.j2 \
    --output-dir slides \
    --TagRemovePreprocessor.remove_input_tags=hide_input
