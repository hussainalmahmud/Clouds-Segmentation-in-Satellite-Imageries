#!/bin/bash
# for CONFIG in config/*.py; do
for CONFIG in config/*.py; do

    if [ "$CONFIG" != "config/__init__.py" ]; then  # Exclude the __init__.py file
        BASENAME=$(basename $CONFIG .py)
        CUDA_VISIBLE_DEVICES=0,1 python main.py --config $BASENAME
    fi
done
