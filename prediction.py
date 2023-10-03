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
from model.unet import UNet
from model.unet_base import UNet_base
from utils.data_utils import LoadDataset, DataTransform
import segmentation_models_pytorch as smp
import numpy as np
import imageio

def gather_image_paths(path_pattern):
    return glob.glob(path_pattern)

def create_dataframe_from_paths(paths):
    return pd.DataFrame(paths, columns=['image_path'])

def gather_image_paths(path_pattern):
    return glob.glob(path_pattern)

def run_inference(df_test, model_path, PATH_OUTPUT, batch_size=4):

    test_loader = DataLoader(df_test, batch_size=batch_size, shuffle=False)
    
    model = UNet_base()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()


    for i, (img) in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            img = img.cuda()
            output = model(img)
            output = output.squeeze(dim=1)
            preds = torch.sigmoid(output)

            for batch_i, pred in enumerate(preds.cpu().numpy()):
                
                df_actual = df_test.get_dataframe()
                PATH = df_actual['image_path'].iloc[i*batch_size + batch_i]
                fname = os.path.basename(PATH)

                # fname_wo_ext = fname.split('.')[0]
                # plt.figure(figsize=(6, 6))
                # plt.title(f'Prediction: {fname}')
                # plt.imshow(pred, cmap='jet', vmin=0, vmax=1)
                # plt.colorbar()
                # plt.savefig(f"{PATH_OUTPUT}/predict_{fname_wo_ext}.png")
                # plt.clf()
                # plt.close()
                
                pred_sub = cv2.resize(pred, (1000, 1000), interpolation=0)
                pred = (pred_sub > 0.5).astype(np.uint8)

                idx = fname.split("_")[-1]
                imageio.imsave(f"{PATH_OUTPUT}/evaluation_mask_{idx}", pred)
                
                
    # zip the file
    import shutil
    PATH_ZIP = f'./OUTPUT_Predictions/submit.zip'
    if os.path.exists(PATH_ZIP):
        os.remove(PATH_ZIP)
    print("zipping...")
    shutil.make_archive(PATH_ZIP[:-4], 'zip', PATH_OUTPUT)
    print("successfully zipped")
                

def main():
    image_paths = gather_image_paths('./data/evaluation_true_color/evaluation_*.tif')
    df_test = create_dataframe_from_paths(image_paths)
    print(f"{len(image_paths)} images")
    print(df_test.head())

    df_test = LoadDataset(df_test, "test", DataTransform())
    
    PATH_OUTPUT = './OUTPUT_Predictions/submission/'
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    # Now run the bone inference
    model_path = "./checkpoints/best_model_unet.pth"
    
    # check if device gpu is 0
    print("torch.cuda.current_device: ",torch.cuda.current_device())  
    run_inference(df_test, model_path=model_path, PATH_OUTPUT=PATH_OUTPUT,batch_size=4)

if __name__ == '__main__':
    main()


    # If the model was trained with DataParallel, the keys will have 'module.' prefix
    # Remove the 'module.' prefix to match the keys with the model's current state_dict
    # model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1)
    # Load the checkpoint
    # checkpoint = torch.load('./checkpoints/model.pth')
    # new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    # model.load_state_dict(new_state_dict)
    
    
    # model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5) # best model so far 0.26
    