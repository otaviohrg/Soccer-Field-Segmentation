import os
import torch
import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2


class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, small_mask=False):
        super(DatasetSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'images', '*'))
        self.mask_files = []
        self.small_mask = small_mask
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path,
                                                'segmentations',
                                                ".".join(os.path.basename(img_path).split(".")[:-1]) + ".png"))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = Image.open(os.path.join(img_path)).convert("RGB")
        mask = Image.open(os.path.join(mask_path)).convert("L")

        i, j, h, w = v2.RandomCrop.get_params(data, output_size=(224, 224))

        randomCrop = lambda x: v2.functional.crop(x, i, j, h, w)

        transform = v2.Compose(
            [
                v2.Resize((356, 356), antialias=True),
                v2.Lambda(randomCrop),
                v2.ToImageTensor(),
                v2.ConvertImageDtype(),
            ]
        )

        transform_data = v2.Compose([
            transform,
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.ToPILImage()
        ])

        transform_mask = v2.Compose([
            transform,
        ])

        if self.small_mask:
            transform_mask = v2.Compose([
                transform_mask,
                v2.Resize((112, 112), antialias=True),
            ])

        mask = transform_mask(mask)
        data = transform_data(data)

        data = np.array(data).transpose(2, 0, 1)
        mask = np.array(mask)

        mask[mask <= 0.1] = 0
        mask[mask >= 0.9] = 0.1
        mask[mask > 0.1] = 0.2
        mask *= 10

        return torch.from_numpy(data).float(), torch.from_numpy(mask).long()

    def __len__(self):
        return len(self.img_files)
