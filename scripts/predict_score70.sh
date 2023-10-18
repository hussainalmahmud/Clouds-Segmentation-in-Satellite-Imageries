#!/bin/bash
IMAGE_PATHS_PATTERN="./data/evaluation_true_color/evaluation_*.tif"
PATH_OUTPUT="./OUTPUT_Predictions/submission/"
BATCH_SIZE=4
INPUT_CHANNELS=3
N_CLASS=1
NETWORKS=" DeepLabV3Plus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus \
UnetPlusPlus 
"

ENCODERS="efficientnet-b3 \
timm-efficientnet-b0 \
timm-efficientnet-b1 \
timm-efficientnet-b3 \
timm-efficientnet-b3 \
timm-efficientnet-b5 \
tu-efficientnetv2_rw_s \
tu-efficientnetv2_rw_s 

"

# CHECKPOINT_PATHS="
# ./checkpoints/pth/best_DeepLabV3Plus_efficientnet-b3fold0.pth \
# ./checkpoints/pth/best_UnetPlusPlus_timm-efficientnet-b0fold1.pth \
# ./checkpoints/pth/best_UnetPlusPlus_timm-efficientnet-b1fold2.pth \
# ./checkpoints/pth/best_UnetPlusPlus_timm-efficientnet-b3fold3.pth \
# ./checkpoints/pth/best_UnetPlusPlus_timm-efficientnet-b3fold0.pth \
# ./checkpoints/pth/best_UnetPlusPlus_timm-efficientnet-b5fold1.pth \
# ./checkpoints/pth/best_UnetPlusPlus_tu-efficientnetv2_rw_sfold3.pth 
# ./checkpoints/pth/best_UnetPlusPlus_tu-efficientnetv2_rw_sfold2.pth \
# "

CHECKPOINT_PATHS="
./checkpoint_naive_70_score/efficientnet-b3_DeepLabV3Plus_fold_2/ckpt/checkpoint-best.pth \
./checkpoint_naive_70_score/timm-efficientnet-b0_UnetPlusPlus_fold_4/ckpt/checkpoint-best.pth \
./checkpoint_naive_70_score/timm-efficientnet-b1_UnetPlusPlus_fold_1/ckpt/checkpoint-best.pth \
./checkpoint_naive_70_score/timm-efficientnet-b3_UnetPlusPlus_fold_3/ckpt/checkpoint-best.pth \
./checkpoint_naive_70_score/timm-efficientnet-b3_UnetPlusPlus_fold_4/ckpt/checkpoint-best.pth \
./checkpoint_naive_70_score/timm-efficientnet-b5_UnetPlusPlus_fold_4/ckpt/checkpoint-best.pth \
./checkpoint_naive_70_score/tu-efficientnetv2_rw_s_UnetPlusPlus_fold_1/ckpt/checkpoint-best.pth \
./checkpoint_naive_70_score/tu-efficientnetv2_rw_s_UnetPlusPlus_fold_4/ckpt/checkpoint-best.pth \
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
