#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python main.py -c DeepLabV3Plus_efb1
CUDA_VISIBLE_DEVICES=0,1 python main.py -c DeepLabV3Plus_efb2
CUDA_VISIBLE_DEVICES=0,1 python main.py -c DeepLabV3Plus_efb3
CUDA_VISIBLE_DEVICES=0,1 python main.py -c DeepLabV3Plus_resnet101
CUDA_VISIBLE_DEVICES=0,1 python main.py -c UnetPlusPlus_efb1
CUDA_VISIBLE_DEVICES=0,1 python main.py -c UnetPlusPlus_efb2
CUDA_VISIBLE_DEVICES=0,1 python main.py -c UnetPlusPlus_efb3
CUDA_VISIBLE_DEVICES=0,1 python main.py -c UnetPlusPlus_resnet101
