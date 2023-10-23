#!/usr/bin/env python3
import os
import torch
import argparse
import shutil
import logging
from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
from torch.utils.data import DataLoader
import cv2
from utils.data_utils import LoadDataset, DataTransform
import imageio
from model.smp_models import SegModel



# def load_model_state_dict(checkpoint_path, use_dataparallel=False):
#     """
#     Load model state dict from a checkpoint file.
    
#     Args:
#         checkpoint_path (str): Path to the checkpoint file.
#         use_dataparallel (bool): If True, prefix keys with 'module.'.

#     Returns:
#         state_dict (OrderedDict): State dictionary with appropriately modified keys.
#     """
#     checkpoint = torch.load(checkpoint_path)
    
#     # Extract the state_dict or the model weights directly, depending on the checkpoint file structure
#     state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
#     # Remove or add 'module.' prefix in the state_dict keys based on the use_dataparallel flag
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key
#         if use_dataparallel and not key.startswith('module.'):
#             new_key = 'module.' + key
#         elif not use_dataparallel and key.startswith('module.'):
#             new_key = key.replace('module.', '', 1)
        
#         # Additional adaptation for 'model.' prefix in the keys
#         new_key = new_key.replace('model.', '', 1)

#         new_state_dict[new_key] = value
    
#     return new_state_dict



def load_model_state_dict(checkpoint_path, model):
    """Load model state dict from a checkpoint file."""
    checkpoints = torch.load(checkpoint_path)
    
    if "state_dict" in checkpoints and checkpoint_path.split(".")[-1] == "pth":
        state_dict = checkpoints["state_dict"]
        
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        if isinstance(model, torch.nn.DataParallel):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = "module." + key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        return state_dict
    
    return checkpoints

# def load_model_state_dict(checkpoint_path):
#     checkpoints = torch.load(checkpoint_path)
#     assert checkpoint_path.split(".")[-1] in ['pth', 'pkl']
#     if checkpoint_path.split(".")[-1] == 'pth':
#         return checkpoints['state_dict']
#     else:
#         return checkpoints

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

    return (
        torch.sigmoid(predict_1) + torch.sigmoid(predict_2) + torch.sigmoid(predict_3)
    ).cpu().numpy().astype("float32") / 3

def apply_morphological_ops(input_mask, kernel_size=5): # increased prediction score by 0.0001
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    refined_mask = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask

def run_inference(models, test_loader, PATH_OUTPUT, df_test,device, batch_size=4):
    """
    Run inference using the provided model(s) and test data loader, saving predictions to the specified output path.

    Parameters:
        models (list): A list of trained models to use for ensemble prediction.
        test_loader (DataLoader): The PyTorch DataLoader to use for test data.
        PATH_OUTPUT (str): Directory path to save prediction images.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 4.
    """

    for idx, model in enumerate(models):
        model = model.to(device).float()  # Ensure model is float32 and on GPU.
        models[idx] = torch.nn.DataParallel(model)  # Wrap model with DataParallel.
        model.eval()  # Set the model to evaluation mode.


    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            # img = data.to(device, dtype=torch.half)
            # img = data.to(device).half()
            # model.float()  # Ensure model is in float32
            img = data.to(device)

            batch_out_all = []

            for j, model in enumerate(models):
                model.eval()
                if j < 6:
                    output = model(img)
                    output = output.squeeze(dim=1)
                    output = torch.sigmoid(output).cpu().numpy().astype("float32")
                else:
                    # output = get_model_output(model, data)
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
                assert index < len(
                    df_actual
                ), f"Index ({index}) out of bounds. i: {i}, batch_i: {batch_i}"

                PATH = df_actual["image_path"].iloc[index]
                fname = os.path.basename(PATH)

                pred_sub = cv2.resize(pred, (1000, 1000), interpolation=0)
                pred_threshold = (pred_sub > 0.5).astype(np.uint8)
                
                pred_threshold = apply_morphological_ops(pred_threshold)

                

                idx = fname.split("_")[-1]
                imageio.imsave(f"{PATH_OUTPUT}/evaluation_mask_{idx}", pred_threshold)

    PATH_ZIP = f"./OUTPUT_Predictions/submit.zip"
    if os.path.exists(PATH_ZIP):
        os.remove(PATH_ZIP)
    print("zipping...")
    shutil.make_archive(PATH_ZIP[:-4], "zip", PATH_OUTPUT)
    print("successfully zipped")


def main(
    image_paths,
    path_output,
    batch_size,
    in_channels,
    n_class,
    networks,
    encoders,
    checkpoint_paths,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    image_paths = gather_image_paths(image_paths)
    df_test = create_dataframe_from_paths(image_paths)

    df_test = LoadDataset(df_test, "test", DataTransform())
    test_loader = DataLoader(df_test, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    PATH_OUTPUT = path_output
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    logging.info(f"Networks: {networks}")
    logging.info(f"Encoders: {encoders}")
    checkpoint_paths = checkpoint_paths
    models = []
    input_channels = in_channels
    num_class = n_class
    use_dataparallel = False
    for i in range(len(networks)):
        Network = networks[i]
        Encoder = encoders[i]
        checkpoint_path = checkpoint_paths[i]
        model = SegModel(
            encoder=Encoder,
            network=Network,
            in_channels=input_channels,
            n_class=num_class,
            pre_train=None,
        ).cuda()
        
        checkpoints = load_model_state_dict(checkpoint_path, use_dataparallel)
        
        model.load_state_dict(checkpoints)
        # model = model.to(device).half()

        models.append(model)
        model = torch.nn.DataParallel(model)

    logging.info(f"Number of Models: {models}")

    run_inference(
        models=models,
        test_loader=test_loader,
        PATH_OUTPUT=PATH_OUTPUT,
        df_test=df_test,
        device= device,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run inference on images.")
    parser.add_argument(
        "--image_paths",
        type=str,
        required=True,
        help="Pattern for gathering image paths",
    )
    parser.add_argument(
        "--path_output", type=str, required=True, help="Path for output"
    )
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--in_channels", type=int, required=True, help="Input channels")
    parser.add_argument("--n_class", type=int, required=True, help="Number of classes")
    parser.add_argument(
        "--networks", type=str, required=True, nargs="+", help="Networks"
    )
    parser.add_argument(
        "--encoders", type=str, required=True, nargs="+", help="Encoders"
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        required=True,
        nargs="+",
        help="Paths to model checkpoint files",
    )

    args = parser.parse_args()

    assert (
        len(args.networks) == len(args.encoders) == len(args.checkpoint_paths)
    ), "Mismatch in the length of network, encoder, and checkpoint path lists"

    main(
        image_paths=args.image_paths,
        path_output=args.path_output,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        n_class=args.n_class,
        networks=args.networks,
        encoders=args.encoders,
        checkpoint_paths=args.checkpoint_paths,
    )


# with torch.no_grad():
#     for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
#         img = data.cuda()
#         batch_out_all = []

#         for j, model in enumerate(models):
#             model.eval()
#             if j < 6:
#                 output = model(img)
#                 output = output.squeeze(dim=1)
#                 output = torch.sigmoid(output).cpu().numpy().astype("float32")
#             else:
#                 # Define data augmentation transformations
#                 augmentation = transforms.Compose([
#                     transforms.RandomHorizontalFlip(),
#                     transforms.RandomVerticalFlip(),
#                     transforms.RandomRotation(90),
#                 ])
                
#                 # Apply data augmentation to the test image
#                 augmented_images = [augmentation(img) for _ in range(3)]  # Apply augmentation 3 times
                
#                 # Convert augmented images to tensors
#                 augmented_image_tensors = [transforms.ToTensor()(aug_img).unsqueeze(0) for aug_img in augmented_images]
                
#                 # Generate predictions for each augmented image
#                 predictions = [model(aug_img).squeeze(dim=1) for aug_img in augmented_image_tensors]
                
#                 # Average the predictions
#                 output = sum(predictions) / len(predictions)
#                 # median the predictions
#                 output = torch.median(torch.stack(predictions), dim=0)[0]
#             batch_out_all.append(output)

#         # Further processing of batch_out_all as needed
