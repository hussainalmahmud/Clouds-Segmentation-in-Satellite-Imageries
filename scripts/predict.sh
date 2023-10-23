#!/bin/bash
IMAGE_PATHS_PATTERN="./data/evaluation_true_color/evaluation_*.tif"
PATH_OUTPUT="./OUTPUT_Predictions/submission/"
BATCH_SIZE=4
INPUT_CHANNELS=4
N_CLASS=1
NETWORKS=" DeepLabV3Plus \
DeepLabV3Plus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus 
"

ENCODERS="efficientnet-b3 \
efficientnet-b4 \
timm-efficientnet-b0 \
timm-efficientnet-b1 \
timm-efficientnet-b3 \
timm-efficientnet-b4 \
timm-efficientnet-b5 \
tu-efficientnetv2_rw_s 
"

CHECKPOINT_PATHS="
./checkpoints/efficientnet-b3_DeepLabV3Plus/pth/best_model.pth \
./checkpoints/efficientnet-b4_DeepLabV3Plus/pth/best_model.pth \
./checkpoints/timm-efficientnet-b0_UnetPlusPlus/pth/best_model.pth \
./checkpoints/timm-efficientnet-b1_UnetPlusPlus/pth/best_model.pth \
./checkpoints/timm-efficientnet-b3_UnetPlusPlus/pth/best_model.pth \
./checkpoints/timm-efficientnet-b4_UnetPlusPlus/pth/best_model.pth \
./checkpoints/timm-efficientnet-b5_UnetPlusPlus/pth/best_model.pth \
./checkpoints/tu-efficientnetv2_rw_s_UnetPlusPlus/pth/best_model.pth \
"

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
