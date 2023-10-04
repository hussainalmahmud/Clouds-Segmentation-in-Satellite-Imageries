#!/bin/bash

IMAGE_PATHS_PATTERN="./data/evaluation_true_color/evaluation_*.tif"
PATH_OUTPUT="./OUTPUT_Predictions/submission/"
BATCH_SIZE=4
NETWORKS="DeepLabV3Plus DeepLabV3Plus DeepLabV3Plus DeepLabV3Plus DeepLabV3Plus"
ENCODERS="timm-efficientnet-b0 timm-efficientnet-b1 timm-efficientnet-b2 timm-efficientnet-b3 timm-efficientnet-b4"
CHECKPOINT_PATHS="./checkpoints/timm-efficientnet-b0_DeepLabV3Plus_fold_0/pth/best_model.pth ./checkpoints/timm-efficientnet-b1_DeepLabV3Plus_fold_1/pth/best_model.pth ./checkpoints/timm-efficientnet-b2_DeepLabV3Plus_fold_2/pth/best_model.pth ./checkpoints/timm-efficientnet-b3_DeepLabV3Plus_fold_3/pth/best_model.pth ./checkpoints/timm-efficientnet-b4_DeepLabV3Plus_fold_4/pth/best_model.pth"

python3 your_script.py $IMAGE_PATHS_PATTERN $PATH_OUTPUT $BATCH_SIZE "$NETWORKS" "$ENCODERS" "$CHECKPOINT_PATHS"



import os
import torch
#... [other imports]

def main(image_paths_pattern, path_output, batch_size, networks, encoders, checkpoint_paths):
    image_paths = gather_image_paths(image_paths_pattern)
    #... [rest of your code]

if __name__ == "__main__":
    # parsing code will be added here


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on images.')
    parser.add_argument('image_paths_pattern', type=str, help='Pattern for gathering image paths')
    parser.add_argument('path_output', type=str, help='Path for output')
    parser.add_argument('batch_size', type=int, help='Batch size')
    parser.add_argument('networks', type=str, nargs='+', help='Networks')
    parser.add_argument('encoders', type=str, nargs='+', help='Encoders')
    parser.add_argument('checkpoint_paths', type=str, nargs='+', help='Paths to model checkpoint files')
    
    args = parser.parse_args()
    main(
        args.image_paths_pattern,
        args.path_output,
        args.batch_size,
        args.networks,
        args.encoders,
        args.checkpoint_paths
    )
