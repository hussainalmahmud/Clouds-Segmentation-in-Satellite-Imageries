#!/usr/bin/env python3
import os
import torch
import cv2
import tifffile
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import cv2
import tifffile
import glob
import pandas as pd
from archive.unet import UNet
from model.unet_base import UNet_base
from utils.data_utils import LoadDataset, DataTransform
import segmentation_models_pytorch as smp
import numpy as np
import imageio
from model.smp_models import SegModel
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

def load_model_state_dict(model, checkpoint_path):
    """Load model state dict from a checkpoint file."""
    checkpoints = torch.load(checkpoint_path)
    if 'state_dict' in checkpoints and checkpoint_path.split(".")[-1] == 'pth':
        state_dict = checkpoints['state_dict']
    else:
        state_dict = checkpoints
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if isinstance(model, torch.nn.DataParallel):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    
def gather_image_paths(path_pattern):
    image_path = glob.glob(path_pattern)
    print(f"{len(image_path)} images")
    return image_path

def create_dataframe_from_paths(paths):
    df_test = pd.DataFrame(paths, columns=["image_path"])
    print(df_test.head())
    return df_test

def run_inference(models, test_loader, PATH_OUTPUT, df_test, batch_size=4):
    """
    Run inference using the provided model(s) and test data loader, saving predictions to the specified output path.

    Parameters:
        models (list): A list of trained models to use for ensemble prediction.
        test_loader (DataLoader): The PyTorch DataLoader to use for test data.
        PATH_OUTPUT (str): Directory path to save prediction images.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 4.
    """
    
    for model in models:
        model.eval()
        model.cuda()

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            img = data.cuda()
            batch_out_all = []
            
            for idx, model in enumerate(models):
                if idx < 6:
                    output = model(img)
                    output = output.squeeze(dim=1)
                    output = torch.sigmoid(output).cpu().numpy().astype('float32')                
                else:
                    print("flip")
                    predict_1 = model(data)[:, 0, :, :]  # [b,c,h,w]->[b,1,h,w]->[b,h,w]
                    predict_2 = model(torch.flip(data, [-1]))[:, 0, :, :]  # [b,c,h,w]->[b,1,h,w]->[b,h,w]
                    predict_2 = torch.flip(predict_2, [-1])
                    predict_3 = model(torch.flip(data, [-2]))[:, 0, :, :]  # [b,c,h,w]->[b,1,h,w]->[b,h,w]
                    predict_3 = torch.flip(predict_3, [-2])
                    output = (torch.sigmoid(predict_1) + torch.sigmoid(predict_2)
                              + torch.sigmoid(predict_3)).cpu().numpy().astype('float32') / 3
                batch_out_all.append(output)
            
            output_mean = np.mean(batch_out_all, 0)
            
            start_index = i * batch_size
            for batch_i, pred in enumerate(output_mean):
                index = start_index + batch_i    
                df_actual = df_test.get_dataframe()
                assert index < len(df_actual), f"Index ({index}) out of bounds. i: {i}, batch_i: {batch_i}"

                PATH = df_actual["image_path"].iloc[index]
                fname = os.path.basename(PATH)

                pred_sub = cv2.resize(pred, (1000, 1000), interpolation=0)
                pred_threshold = (pred_sub > 0.5).astype(np.uint8)

                idx = fname.split("_")[-1]
                imageio.imsave(f"{PATH_OUTPUT}/evaluation_mask_{idx}", pred_threshold)

    PATH_ZIP = f"./OUTPUT_Predictions/submit.zip"
    if os.path.exists(PATH_ZIP):
        os.remove(PATH_ZIP)
    print("zipping...")
    shutil.make_archive(PATH_ZIP[:-4], "zip", PATH_OUTPUT)
    print("successfully zipped")


def main():
    batch_size = 4
    image_paths = gather_image_paths("./data/evaluation_true_color/evaluation_*.tif")
    df_test = create_dataframe_from_paths(image_paths)

    df_test = LoadDataset(df_test, "test", DataTransform())
    test_loader = DataLoader(df_test, batch_size, shuffle=False)

    PATH_OUTPUT = "./OUTPUT_Predictions/submission/"
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    num_workers= os.cpu_count()
    networks = [
        'DeepLabV3Plus',
        'DeepLabV3Plus',
        'DeepLabV3Plus',
        'DeepLabV3Plus',
        'DeepLabV3Plus',
        'DeepLabV3Plus',
        'UnetPlusPlus',
        'UnetPlusPlus',
        'UnetPlusPlus',
        'UnetPlusPlus',]
    encoders = [
                'timm-efficientnet-b0',
                'timm-efficientnet-b1',
                'timm-efficientnet-b2', 
                'timm-efficientnet-b3',
                'timm-efficientnet-b4',
                'resnet101',
                'timm-efficientnet-b1',
                'timm-efficientnet-b2', 
                'timm-efficientnet-b3',
                'resnet101'
                ] 
    # encoders = ['timm-efficientnet-b0','timm-efficientnet-b1','timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4']
    
    checkpoint_paths = [
        "./checkpoints/timm-efficientnet-b0_DeepLabV3Plus/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b1_DeepLabV3Plus/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b2_DeepLabV3Plus/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b3_DeepLabV3Plus/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b4_DeepLabV3Plus/pth/best_model.pth",
        "./checkpoints/resnet101_DeepLabV3Plus/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b1_UnetPlusPlus/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b2_UnetPlusPlus/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b3_UnetPlusPlus/pth/best_model.pth",
        "./checkpoints/resnet101_UnetPlusPlus/pth/best_model.pth",
    ]
    models = []
    in_channels = 3
    n_class = 1
    for i in range(len(networks)):
        network = networks[i]
        encoder = encoders[i]
        checkpoint_path = checkpoint_paths[i]
        model = SegModel(encoder=encoder, network=network,
                           in_channels=in_channels, n_class=n_class,pre_train=None).cuda()
        model = torch.nn.DataParallel(model)
        load_model_state_dict(model, checkpoint_path)
        models.append(model)

    print(f"Loaded {len(models)} models")

    run_inference(
    models=models,
    test_loader=test_loader, 
    PATH_OUTPUT=PATH_OUTPUT,    df_test=df_test, 
    batch_size=batch_size,  
    )

if __name__ == "__main__":
    main()




# import torch
# from torch.nn.functional import sigmoid
# from tqdm import tqdm
# import numpy as np

# def get_flipped_predictions(model, data, flip_dim):
#     flipped_data = torch.flip(data, [flip_dim])
#     predict = model(flipped_data)[:, 0, :, :]
#     return torch.flip(predict, [flip_dim])

# def get_model_output(model, data):
#     predict_1 = model(data)[:, 0, :, :]
#     predict_2 = get_flipped_predictions(model, data, -1)
#     predict_3 = get_flipped_predictions(model, data, -2)

#     return (sigmoid(predict_1) + 
#             sigmoid(predict_2) + 
#             sigmoid(predict_3)).cpu().numpy().astype('float32') / 3

# def process_data(test_loader, models):
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
#             img = data.cuda()
#             batch_out_all = []

#             for idx, model in enumerate(models):
#                 if idx < 4:
#                     output = model(img).squeeze(dim=1)
#                     output = sigmoid(output).cpu().numpy().astype('float32')
#                 else:
#                     print("flip")
#                     output = get_model_output(model, data)
#                 batch_out_all.append(output)
