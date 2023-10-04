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
import time
from time import strftime, localtime

# def load_model_state_dict(checkpoint_path):
#     checkpoints = torch.load(checkpoint_path)
#     # assert checkpoint_path.split(".")[-1] in ['pth', 'pkl']
#     if checkpoint_path.split(".")[-1] == 'pth':
#         return checkpoints['state_dict']
#     else:
#         return checkpoints
def load_model_state_dict(model, checkpoint_path):
    checkpoints = torch.load(checkpoint_path)

    # Check if 'state_dict' key exists and whether it's a .pth file
    if 'state_dict' in checkpoints and checkpoint_path.split(".")[-1] == 'pth':
        state_dict = checkpoints['state_dict']
    else:
        state_dict = checkpoints
    
    # Check if the state_dict keys are saved with 'module.' due to DataParallel
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    # Check if model is instance of nn.DataParallel
    if isinstance(model, torch.nn.DataParallel):
        # Add 'module.' prefix
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


# def run_inference(models, df_test, test_loader, model_path, PATH_OUTPUT, batch_size=4):
"""
Run inference using the provided model(s) and test data loader, saving predictions to the specified output path.

Parameters:
    models (list): A list of trained models to use for ensemble prediction.
    test_loader (DataLoader): The PyTorch DataLoader to use for test data.
    PATH_OUTPUT (str): Directory path to save prediction images.
    batch_size (int, optional): Batch size for the DataLoader. Defaults to 4.
"""
def run_inference(models, test_loader, PATH_OUTPUT, df_test, batch_size=4):

    
    # Ensure all models are in evaluation mode and transferred to GPU
    for model in models:
        model.eval()
        model.cuda()

    # Perform inference
    with torch.no_grad():
        #print test_loader length
        print("test_loader length: ", len(test_loader))
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            img = data.cuda()
            batch_out_all = []
            
            for idx, model in enumerate(models):
                output = model(img)
                output = output.squeeze(dim=1)
                print("output;:: ",output.shape)
                output = torch.sigmoid(output).cpu().numpy().astype('float32')                
                batch_out_all.append(output)
            
            # Mean over all model predictions
            output_mean = np.mean(batch_out_all, 0)
            
            # Calculate the starting index for this batch
            start_index = i * batch_size
            for batch_i, pred in enumerate(output_mean):
                index = start_index + batch_i    
                df_actual = df_test.get_dataframe()
                assert index < len(df_actual), f"Index ({index}) out of bounds. i: {i}, batch_i: {batch_i}"

                PATH = df_actual["image_path"].iloc[index]
                # PATH = df_actual["image_path"].iloc[i * batch_size + batch_i]
                fname = os.path.basename(PATH)

                pred_sub = cv2.resize(pred, (1000, 1000), interpolation=0)
                pred_threshold = (pred_sub > 0.5).astype(np.uint8)

                idx = fname.split("_")[-1]
                imageio.imsave(f"{PATH_OUTPUT}/evaluation_mask_{idx}", pred_threshold)
                # else:
                #     print(f"Index {index} is out of bounds.")

    # zip the file
    import shutil

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


        time_start = time.time()
    print(strftime('%Y-%m-%d %H:%M:%S', localtime()))
    batch_size=4# resnet50,1_5fold,gpu,6245M;efb1,1_5fold,gpu,4173M;
    num_workers=12
    #data path
    root_dir = Path("/codeexecution")
    networks = [
        'DeepLabV3Plus' ] * 4 
    encoders = [
        'timm-efficientnet-b0', 
        'timm-efficientnet-b1',
        'timm-efficientnet-b2', 
        # 'timm-efficientnet-b3',
        'timm-efficientnet-b4']
    checkpoint_paths = [
        "./checkpoints/timm-efficientnet-b0_DeepLabV3Plus_fold_0/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b1_DeepLabV3Plus_fold_1/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b2_DeepLabV3Plus_fold_2/pth/best_model.pth",
        # "./checkpoints/timm-efficientnet-b3_DeepLabV3Plus_fold_3/pth/best_model.pth",
        "./checkpoints/timm-efficientnet-b4_DeepLabV3Plus_fold_4/pth/best_model.pth",
        
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



    run_inference(
    models=models,
    test_loader=test_loader, 
    PATH_OUTPUT=PATH_OUTPUT,    df_test=df_test, 
    batch_size=batch_size,  
    )

if __name__ == "__main__":
    main()
