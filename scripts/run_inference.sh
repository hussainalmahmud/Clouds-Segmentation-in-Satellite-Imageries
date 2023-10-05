#!/bin/bash
IMAGE_PATHS_PATTERN="./data/evaluation_true_color/evaluation_*.tif"
PATH_OUTPUT="./OUTPUT_Predictions/submission/"
BATCH_SIZE=4
INPUT_CHANNELS=3
N_CLASS=1
NETWORKS="DeepLabV3Plus DeepLabV3Plus DeepLabV3Plus DeepLabV3Plus UnetPlusPlus UnetPlusPlus UnetPlusPlus UnetPlusPlus"
ENCODERS="timm-efficientnet-b1 timm-efficientnet-b2 timm-efficientnet-b3 resnet101 timm-efficientnet-b1 timm-efficientnet-b2 timm-efficientnet-b3 resnet101"
CHECKPOINT_PATHS="./checkpoints/timm-efficientnet-b1_DeepLabV3Plus/pth/best_model.pth ./checkpoints/timm-efficientnet-b2_DeepLabV3Plus/pth/best_model.pth ./checkpoints/timm-efficientnet-b3_DeepLabV3Plus/pth/best_model.pth ./checkpoints/resnet101_DeepLabV3Plus/pth/best_model.pth ./checkpoints/timm-efficientnet-b1_UnetPlusPlus/pth/best_model.pth ./checkpoints/timm-efficientnet-b2_UnetPlusPlus/pth/best_model.pth ./checkpoints/timm-efficientnet-b3_UnetPlusPlus/pth/best_model.pth ./checkpoints/resnet101_UnetPlusPlus/pth/best_model.pth"

# Running Python Script
python3 prediction.py \
    --image_paths "$IMAGE_PATHS_PATTERN" \
    --path_output "$PATH_OUTPUT" \
    --batch_size "$BATCH_SIZE" \
    --in_channels "$INPUT_CHANNELS" \
    --n_class "$N_CLASS" \
    --networks $NETWORKS \
    --encoders $ENCODERS \
    --checkpoint_paths $CHECKPOINT_PATHS
