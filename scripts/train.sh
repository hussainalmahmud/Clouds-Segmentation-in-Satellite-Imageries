#!/bin/bash

# First set of configurations
# CONFIGS=("DeepLabV3Plus_efb3" "UnetPlusPlus_efb0" "UnetPlusPlus_efb3" "UnetPlusPlus_efb3")
NET_CONFIG="net_config"

CUDA_VISIBLE_DEVICES=0,1 python main.py -c config
