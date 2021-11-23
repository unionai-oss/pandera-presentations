#!/bin/bash

notebook_path=$1
jupytext --sync $notebook_path
jupyter nbconvert $notebook_path \
    --execute \
    --to=slides \
    --SlidesExporter.reveal_theme=simple \
    --SlidesExporter.reveal_transition=none \
    --SlidesExporter.reveal_scroll=True \
    --template-file .jupyter/slide_template.html.j2 \
    --output-dir=slides \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_input_tags=hide_input \
    --TagRemovePreprocessor.remove_single_output_tags=hide_output \
    --TagRemovePreprocessor.remove_all_outputs_tags=hide_output \
    --CSSHTMLHeaderPreprocessor.enabled=True \
    --CSSHTMLHeaderPreprocessor.style=native
