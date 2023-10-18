import os

config = dict(
    encoder='timm-efficientnet-b3',
    model_network='DeepLabV3Plus',
    in_channels=3,
    n_class=1,
    save_path='',
    learning_rate=0.0003,
    weight_decay=0.0005,
    seed=10000,
)

CONFIGS = {
    
    0: {"encoder": "efficientnet-b3", "model_network": "DeepLabV3Plus"},
    1: {"encoder": "efficientnet-b4", "model_network": "DeepLabV3Plus"},
    2: {"encoder": "timm-efficientnet-b0", "model_network": "UnetPlusPlus"},
    3: {"encoder": "timm-efficientnet-b1", "model_network": "UnetPlusPlus"},
    4: {"encoder": "timm-efficientnet-b3", "model_network": "UnetPlusPlus"},
    5: {"encoder": "timm-efficientnet-b4", "model_network": "UnetPlusPlus"},
    6: {"encoder": "timm-efficientnet-b5", "model_network": "UnetPlusPlus"},
    7: {"encoder": "tu-efficientnetv2_rw_s", "model_network": "UnetPlusPlus"},
}


config_directory = "config"

if not os.path.exists(config_directory):
    os.makedirs(config_directory)

for index, conf_override in CONFIGS.items():
    current_config = dict(config)
    current_config.update(conf_override)

    # Generate file name based on model_network and encoder
    file_name = f"{current_config['model_network']}_{current_config['encoder']}.py"
    file_name = file_name.replace("-", "_")  # replace hyphens with underscores for valid module names

    # Write this config to a new Python file
    with open(os.path.join(config_directory, file_name), 'w') as f:
        f.write("config = " + str(current_config))