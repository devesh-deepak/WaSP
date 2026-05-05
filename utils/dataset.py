import torch
# import os
# import glob
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms.functional as TF
# import random
# from config import Config

# class SRDataset(Dataset):
#     def __init__(self, hr_dir, crop_size=None, upscale_factor=4, augment=True):
#         """
#         Args:
#             hr_dir: Path to High Res images.
#             crop_size: Size of the HR crop (e.g., 128x128). If None, returns full image.
#             augment: Apply random flips/rotations.
#         """
#         self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
#         self.crop_size = crop_size
#         self.scale = upscale_factor
#         self.augment = augment
        
#         if len(self.hr_files) == 0:
#             raise ValueError(f"No .png images found in {hr_dir}. Check the path.")

#     def __len__(self):
#         return len(self.hr_files)

#     def __getitem__(self, idx):
#         # 1. Load HR Image
#         hr_img = Image.open(self.hr_files[idx]).convert("RGB")

#         # 2. Random Crop (only during training)
#         if self.crop_size is not None:
#             # Ensure image is large enough to crop
#             w, h = hr_img.size
#             if w < self.crop_size or h < self.crop_size:
#                 # If image is too small, resize it up slightly or skip cropping logic carefully
#                 # For DIV2K, images are 2K resolution, so this is safe.
#                 pass
#             else:
#                 # Get random crop parameters
#                 i, j, h, w = start_i, start_j, th, tw = \
#                     self.get_random_crop_coords(hr_img, self.crop_size, self.crop_size)
#                 hr_img = TF.crop(hr_img, i, j, h, w)

#         # 3. Data Augmentation (Random Horizontal Flip / Rotation)
#         if self.augment:
#             if random.random() > 0.5:
#                 hr_img = TF.hflip(hr_img)
#             if random.random() > 0.5:
#                 hr_img = TF.vflip(hr_img)

#         # 4. Generate LR Image (On-the-fly Downsampling)
#         # Calculate LR dimensions
#         lr_w = hr_img.size[0] // self.scale
#         lr_h = hr_img.size[1] // self.scale
        
#         # Resize to LR (Bicubic)
#         lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)

#         # 5. Convert to Tensor
#         hr_tensor = TF.to_tensor(hr_img)
#         lr_tensor = TF.to_tensor(lr_img)

#         return lr_tensor, hr_tensor

#     def get_random_crop_coords(self, img, crop_h, crop_w):
#         w, h = img.size
#         i = random.randint(0, h - crop_h)
#         j = random.randint(0, w - crop_w)
#         return i, j, crop_h, crop_w

# def get_loader(is_train=True):
#     if is_train:
#         # Training: Random Crops + Augmentation
#         ds = SRDataset(
#             Config.HR_DIR, 
#             crop_size=Config.CROP_SIZE, 
#             upscale_factor=Config.UPSCALE_FACTOR, 
#             augment=True
#         )
#         return torch.utils.data.DataLoader(
#             ds, 
#             batch_size=Config.BATCH_SIZE, 
#             shuffle=True, 
#             num_workers=Config.NUM_WORKERS,
#             pin_memory=True
#         )
#     else:
#         # Validation: No Cropping (Full Images), No Augmentation
#         # Note: Batch size must be 1 for validation because full images have different sizes
#         ds = SRDataset(
#             Config.HR_DIR, 
#             crop_size=None, 
#             upscale_factor=Config.UPSCALE_FACTOR, 
#             augment=False
#         )
#         return torch.utils.data.DataLoader(
#             ds, 
#             batch_size=1, 
#             shuffle=False, 
#             num_workers=1,
#             pin_memory=True
#         )

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
from config import Config

class SRDataset(Dataset):
    def __init__(self, hr_dir, crop_size=None, upscale_factor=4, augment=True):
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.crop_size = crop_size
        self.scale = upscale_factor
        self.augment = augment
        if len(self.hr_files) == 0: raise ValueError(f"No images in {hr_dir}")

    def __len__(self): return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_files[idx]).convert("RGB")
        
        # Training: Random Crop
        if self.crop_size is not None:
            w, h = hr_img.size
            if w >= self.crop_size and h >= self.crop_size:
                i, j, h_c, w_c = self.get_random_crop_coords(hr_img, self.crop_size, self.crop_size)
                hr_img = TF.crop(hr_img, i, j, h_c, w_c)

        # Training: Augmentation
        if self.augment:
            if random.random() > 0.5: hr_img = TF.hflip(hr_img)
            if random.random() > 0.5: hr_img = TF.vflip(hr_img)

        # Generate LR
        lr_w, lr_h = hr_img.size[0] // self.scale, hr_img.size[1] // self.scale
        lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)

        return TF.to_tensor(lr_img), TF.to_tensor(hr_img)

    def get_random_crop_coords(self, img, crop_h, crop_w):
        w, h = img.size
        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        return i, j, crop_h, crop_w

def get_loader(is_train=True):
    if is_train:
        ds = SRDataset(Config.HR_DIR, crop_size=Config.CROP_SIZE, upscale_factor=Config.UPSCALE_FACTOR, augment=True)
        return torch.utils.data.DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    else:
        # Validation on first 20 images
        ds = SRDataset(Config.HR_DIR, crop_size=None, upscale_factor=Config.UPSCALE_FACTOR, augment=False)
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)