import re
import os
import glob
import pandas as pd
import numpy as np
import tifffile
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

def prepare_full_dataset():
    """
    Prepare training and validation datasets based on the specified fold.

    Args:
        fold (int): The fold number. For fold = 0, the first `fold_size` samples are used for validation.
        num_folds (int): The total number of folds to divide the data.

    Returns:
        train_set, valid_set: The prepared training and validation datasets for the specified fold.
    """

    PATH_IMGS = glob.glob("./data/train_true_color/train_true_color_*.tif")
    PATH_MASKS = glob.glob("./data/train_mask/train_mask_*.tif")

    # Function to extract the numerical identifier from filename
    def extract_number(filepath):
        match = re.search(r"(\d+)", filepath)
        if match:
            return int(match.group(1))
        return -1  

    PATH_IMGS = sorted(PATH_IMGS, key=extract_number)
    PATH_MASKS = sorted(PATH_MASKS, key=extract_number)

    df_dataset = pd.DataFrame({"image_path": PATH_IMGS, "mask_path": PATH_MASKS})

    for _, row in df_dataset.iterrows():
        img_name = os.path.splitext(os.path.basename(row["image_path"]))[0].replace(
            "train_true_color_", ""
        )
        mask_name = os.path.splitext(os.path.basename(row["mask_path"]))[0].replace(
            "train_mask_", ""
        )

        assert img_name == mask_name, f"Mismatch found: {img_name} and {mask_name}"

    
    # df_dataset = df_dataset.iloc[:100] # get only 100 data for only when debugging
    
    print(df_dataset.head())
    print("dataset shape ", df_dataset.shape)

    print("df_dataset set size: ", len(df_dataset))

    
    
    return df_dataset


def prepare_train_valid_dataset():
    """
    Prepare training and validation datasets based on the specified fold.

    Args:
        fold (int): The fold number. For fold = 0, the first `fold_size` samples are used for validation.
        num_folds (int): The total number of folds to divide the data.

    Returns:
        train_set, valid_set: The prepared training and validation datasets for the specified fold.
    """

    PATH_IMGS = glob.glob("./data/train_true_color/train_true_color_*.tif")
    PATH_MASKS = glob.glob("./data/train_mask/train_mask_*.tif")
    PATH_SHADOW_MASKS = glob.glob("./data/shadow_mask/shadow_mask_*.tif")

    # Function to extract the numerical identifier from filename
    def extract_number(filepath):
        match = re.search(r"(\d+)", filepath)
        if match:
            return int(match.group(1))
        return -1  

    PATH_IMGS = sorted(PATH_IMGS, key=extract_number)
    PATH_MASKS = sorted(PATH_MASKS, key=extract_number)
    PATH_SHADOW_MASKS = sorted(PATH_SHADOW_MASKS, key=extract_number)

    print(len(PATH_IMGS))
    print(len(PATH_MASKS))
    print(len(PATH_SHADOW_MASKS))

    dataset = pd.DataFrame({"image_path": PATH_IMGS, "mask_path": PATH_MASKS, "shadow_mask_path": PATH_SHADOW_MASKS})

    for _, row in dataset.iterrows():
        img_name = os.path.splitext(os.path.basename(row["image_path"]))[0].replace(
            "train_true_color_", ""
        )
        mask_name = os.path.splitext(os.path.basename(row["mask_path"]))[0].replace(
            "train_mask_", ""
        )
        shadow_name = os.path.splitext(os.path.basename(row["shadow_mask_path"]))[0].replace(
            "shadow_mask_", ""
        )

        # assert  for img_name, mask_namd and shadow_name
        assert img_name == mask_name, f"Mismatch found: {img_name} and {mask_name}"
        assert mask_name == shadow_name, f"Mismatch found: {mask_name} and {shadow_name}" 

    # dataset = dataset.iloc[:100] # get only 100 data for only when debugging
    print(dataset.head())
    print("dataset shape ", dataset.shape)

    train_df, valid_df = train_test_split(dataset, test_size=0.2, random_state=42)

    train_set = LoadDataset(train_df, "train", transform=DataTransform())
    valid_set = LoadDataset(valid_df, "valid", transform=DataTransform())

    print("Training set size: ", len(train_set))
    print("Validation set size: ", len(valid_set))

    return train_set, valid_set


class DataTransform:
    """
    A class used to transform datasets.

    Attributes:
            data_transform (dict): Contains the transformation operations for different phases (e.g. "train").

    Methods:
            __call__(phase, img, mask=None): Transforms the given image and mask based on the specified phase.

    Usage:
            transform = DataTransform()
            transformed_data = transform("train", img, mask)
    """

    def __init__(self):
        self.data_transform = {
            "train": A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.0625,rotate_limit=15,p=0.5), 
                    A.GridDistortion(p=0.35), 
                    A.GaussianBlur(p=0.25), 
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomResizedCrop(512, 512, interpolation=1, p=1),
                    ToTensorV2(),
                ]
            ),
            "valid": A.Compose(
                [
                    A.Resize(512, 512, interpolation=1, p=1),
                    ToTensorV2(),
                ]
            ),
            "test": A.Compose(
                [
                    A.Resize(512, 512, interpolation=1, p=1),
                    ToTensorV2(),
                ]
            ),
        }

    def __call__(self, phase, img, mask=None):
        if mask is None:
            return self.data_transform[phase](image=img)
        else:
            transformed = self.data_transform[phase](image=img, mask=mask)
            return transformed


class LoadDataset(Dataset):
    def __init__(self, df, phase, transform):
        """
        Args:
            df (DataFrame): DataFrame containing file paths. Must have 'image_path' column. If masks are present, it should also have 'mask_path'.
            phase (str): One of train, valid, test.
            transform (DataTransform): Transformation class.
        """
        self.df = df
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def get_dataframe(self):
        return self.df

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = row["image_path"]
        img = tifffile.imread(img_path).astype(np.float32)
        img = np.clip(img, 400, 2400) / 2400

        shadow_mask_path = row["shadow_mask_path"]
        shadow_mask = tifffile.imread(shadow_mask_path).astype(np.float32)
        
        # Concatenate shadow mask along the channel dimension
        img = np.concatenate([img, shadow_mask[:, :, np.newaxis]], axis=-1)

        if self.phase == "test":
            transformed = self.transform(self.phase, img)
            return transformed["image"]
        else:
            mask_path = row["mask_path"]
            mask = tifffile.imread(mask_path).astype(np.float32)
            transformed = self.transform(self.phase, img, mask)
            
            return transformed["image"], transformed["mask"]