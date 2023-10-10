# net_config = {
#     0: {"encoder": "efficientnet-b3", "model_network": "DeepLabV3Plus"},
#     1: {"encoder": "timm-efficientnet-b0", "model_network": "UnetPlusPlus"},
#     2: {"encoder": "timm-efficientnet-b1", "model_network": "UnetPlusPlus"},
#     3: {"encoder": "timm-efficientnet-b3", "model_network": "UnetPlusPlus"},
# }

net_config = {
    0: {"encoder": "timm-efficientnet-b3", "model_network": "UnetPlusPlus"},
    1: {"encoder": "timm-efficientnet-b5", "model_network": "UnetPlusPlus"},
    2: {"encoder": "tu-efficientnetv2_rw_s", "model_network": "UnetPlusPlus"},
    3: {"encoder": "tu-efficientnetv2_rw_s", "model_network": "UnetPlusPlus"},
}

config = dict(
    in_channels=3,
    n_class=1,
    save_path='',
    learning_rate = 0.0001,
    weight_decay = 0.0005,
    seed=10000,
)
def generate_save_path(index):
    try:
        # Use global keyword to reference the variable defined outside of the function
        global net_config
        current_config = net_config[index]
        return '{}_{}_{}'.format(current_config['model_network'], current_config['encoder'], index)
    except KeyError:
        # Handle error if index doesn’t exist in network_configs
        print(f"Index {index} not found in network_configs.")
        return None
