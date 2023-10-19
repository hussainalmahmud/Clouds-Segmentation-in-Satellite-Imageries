for CONFIG in config/UnetPlusPlus_timm_efficientnet_b4.py; do
    if [ "$CONFIG" != "config/__init__.py" ]; then  # Exclude the __init__.py file
        BASENAME=$(basename $CONFIG .py)
        CUDA_VISIBLE_DEVICES=0,1 python main.py --config $BASENAME
    fi
done
