import torch
import glob
import os
from PIL import Image

from torch.utils.data import Dataset
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2


class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, small_mask=False):
        super(DatasetSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'images', '*'))
        self.mask_files = [
            os.path.join(folder_path, 'segmentations',
                         ".".join(os.path.basename(p).split(".")[:-1]) + ".png")
            for p in self.img_files
        ]
        self.small_mask = small_mask

    def __getitem__(self, index):
        data = Image.open(self.img_files[index]).convert("RGB")
        mask = Image.open(self.mask_files[index]).convert("L")

        i, j, h, w = v2.RandomCrop.get_params(data, output_size=(224, 224))
        crop = lambda x: v2.functional.crop(x, i, j, h, w)

        spatial = v2.Compose([
            v2.Resize((356, 356), antialias=True),
            v2.Lambda(crop),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        data = v2.Compose([
            spatial,
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(data)  # (3, 224, 224) float32, ImageNet-normalised

        mask = spatial(mask)  # (1, 224, 224) float32 in [0, 1]

        if self.small_mask:
            mask = v2.functional.resize(mask, (112, 112), antialias=True)

        # Map grayscale intensity to class indices:
        #   ~0   (black, background) → 0
        #   ~0.5 (gray, lines)       → 1
        #   ~1.0 (white, field)      → 2
        mask = mask.squeeze(0).numpy()
        mask[mask <= 0.1] = 0
        mask[(mask > 0.1) & (mask < 0.9)] = 0.1
        mask[mask >= 0.9] = 0.2
        mask *= 10
        mask = torch.from_numpy(mask).long()

        return data, mask.unsqueeze(0)

    def __len__(self):
        return len(self.img_files)
