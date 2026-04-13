import torch
from torch.utils.data import DataLoader, random_split

from soccer_segmentation.data.dataloader.dataset import DatasetSegmentation


def get_loader(folder_path, shuffle, small_mask=False, batch_size=32, num_workers=8, pin_memory=True):
    dataset = DatasetSegmentation(folder_path=folder_path, small_mask=small_mask)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_train_val_loaders(folder_path, val_size, seed, small_mask=False, batch_size=32, num_workers=8, pin_memory=True):
    """Load the train folder and split into train/val subsets."""
    dataset = DatasetSegmentation(folder_path=folder_path, small_mask=small_mask)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(f"val_size={val_size} is too large for a dataset of {len(dataset)} images")
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
