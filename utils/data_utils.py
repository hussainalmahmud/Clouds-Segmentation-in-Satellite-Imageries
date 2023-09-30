import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from albumentations import Compose
import albumentations as A
from skimage.transform import resize


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', augment: bool = False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.mask_ids = [splitext(file)[0] for file in listdir(mask_dir) if isfile(join(mask_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        self.augment = augment
        if self.augment:
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                # Add more augmentations as needed
            ])
        
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        self.mask_values = [0, 1]
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, img_array, scale, is_mask):
        
        w, h = img_array.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        if is_mask:
            img_resized = resize(img_array, (newH, newW), order=0, preserve_range=True, mode='constant')
        else:
            img_resized = resize(img_array, (newH, newW), order=3, preserve_range=True, mode='constant')
        
        # Ensure we return the same dtype as the input
        img_resized = img_resized.astype(img_array.dtype)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img_resized.ndim == 2:
                    mask[img_resized == v] = i
                else:
                    mask[(img_resized == v).all(-1)] = i

            return mask

        else:
            if img_resized.ndim == 2:
                img_resized = img_resized[np.newaxis, ...]
            else:
                img_resized = img_resized.transpose((2, 0, 1))

            if (img_resized > 1).any():
                img_resized = img_resized / 255.0

            return img_resized

    def __getitem__(self, idx):
        img_path = self.ids[idx]
        mask_path = self.mask_ids[idx]
        images_dir = self.images_dir
        mask_dir = self.mask_dir

        
        imgs_file = join(images_dir, img_path + '.tif')
        masks_file = join(mask_dir, mask_path + '.tif')
        import tifffile
        img = tifffile.imread(imgs_file).astype(np.float32)
        mask = tifffile.imread(masks_file).astype(np.float32)

        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {img_path} should be the same size, but are {img.shape} and {mask.shape}'

        if self.augment:
            augmented = self.augmentations(image=np.array(img), mask=np.array(mask))
            img = augmented['image']
            mask = augmented['mask']
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CloudDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')