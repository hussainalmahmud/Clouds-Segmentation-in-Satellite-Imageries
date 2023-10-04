#!/usr/bin/env python3
import os
import torch
import cv2
from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import cv2
import glob
import pandas as pd
from model.unet_base import UNet_base
from utils.data_utils import LoadDataset, DataTransform
import imageio
from model.smp_models import SegModel
import cv2
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

def get_flipped_predictions(model, data, flip_dim):
    flipped_data = torch.flip(data, [flip_dim])
    predict = model(flipped_data).squeeze(dim=1) 
    return torch.flip(predict, [flip_dim])

def get_model_output(model, data):
    predict_1 = model(data).squeeze(dim=1) 
    predict_2 = get_flipped_predictions(model, data, -1)
    predict_3 = get_flipped_predictions(model, data, -2)

    return (torch.sigmoid(predict_1) + 
            torch.sigmoid(predict_2) + 
            torch.sigmoid(predict_3)).cpu().numpy().astype('float32') / 3

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
                    output = get_model_output(model, data)
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


def main(image_paths, path_output, batch_size, networks, encoders, checkpoint_paths):
    batch_size = batch_size
    image_paths = gather_image_paths(image_paths)
    df_test = create_dataframe_from_paths(image_paths)

    df_test = LoadDataset(df_test, "test", DataTransform())
    test_loader = DataLoader(df_test, batch_size, shuffle=False)

    PATH_OUTPUT = path_output
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    num_workers= os.cpu_count()
    networks = networks
    
    encoders = encoders
    print(type(encoders), encoders)
    checkpoint_paths = checkpoint_paths
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
    PATH_OUTPUT=PATH_OUTPUT, df_test=df_test, 
    batch_size=batch_size,  
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on images.')
    parser.add_argument('--image_paths', type=str, required=True, help='Pattern for gathering image paths')
    parser.add_argument('--path_output', type=str, required=True, help='Path for output')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--networks', type=str, required=True, nargs='+', help='Networks')
    parser.add_argument('--encoders', type=str, required=True, nargs='+', help='Encoders')
    parser.add_argument('--checkpoint_paths', type=str, required=True, nargs='+', help='Paths to model checkpoint files')
    
    args = parser.parse_args()
    
    assert len(args.networks) == len(args.encoders) == len(args.checkpoint_paths), "Mismatch in the length of network, encoder, and checkpoint path lists"

    # Run main function with parsed arguments
    main(
        image_paths=args.image_paths,
        path_output=args.path_output,
        batch_size=args.batch_size,
        networks=args.networks,
        encoders=args.encoders,
        checkpoint_paths=args.checkpoint_paths
    )
    