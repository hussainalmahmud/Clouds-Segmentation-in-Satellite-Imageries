#!/bin/bash
# CONFIGS=("DeepLabV3Plus_efb1" "DeepLabV3Plus_efb2" "DeepLabV3Plus_efb3" "DeepLabV3Plus_resnet101"
#          "UnetPlusPlus_efb1" "UnetPlusPlus_efb2" "UnetPlusPlus_efb3" "UnetPlusPlus_resnet101")

# CONFIGS=(encode"DeepLabV3Plus_efb1" "DeepLabV3Plus_efb2" "DeepLabV3Plus_efb3" "DeepLabV3Plus_resnet101"
#          "UnetPlusPlus_efb1" "UnetPlusPlus_efb2" "UnetPlusPlus_efb3" "UnetPlusPlus_resnet101")
# for CONFIG in "${CONFIGS[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1 python main.py -c $CONFIG
# done


for CONFIG in config/UnetPlusPlus_*.py; do
    if [ "$CONFIG" != "config/__init__.py" ]; then  # Exclude the __init__.py file
        BASENAME=$(basename $CONFIG .py)
        CUDA_VISIBLE_DEVICES=0,1 python main.py --config $BASENAME
    fi
done
