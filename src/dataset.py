import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class LandCoverDataset(Dataset):
    def __init__(self, images_dir, masks_dir, patch_size=512):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.patch_size = patch_size

        self.image_files = sorted(os.listdir(images_dir))
        self.masks_files = sorted(os.listdir(masks_dir))

        assert len(self.image_files) == len(self.masks_files), \
            "NUMBER OF IMAGES AND MASKS MUST MATCH "
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_files[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        h, w, _ = image.shape
        ps = self.patch_size

        # random patch coords
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        image_patch = image[y:y+ps, x:x+ps]
        mask_patch = mask[y:y+ps, x:x+ps]

        # normalization
        image_patch = image_patch.astype(np.float32) / 255

        # converting to tensors using pytorch
        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1)
        mask_patch = torch.from_numpy(mask_patch).long()

        return image_patch, mask_patch