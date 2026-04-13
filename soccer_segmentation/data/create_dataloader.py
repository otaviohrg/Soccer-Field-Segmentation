from torch.utils.data import DataLoader

from soccer_segmentation.data.dataloader.dataset import DatasetSegmentation


def get_loader(folder_path, shuffle, small_mask=False, batch_size=32, num_workers=8, pin_memory=True):
    dataset = DatasetSegmentation(folder_path=folder_path, small_mask=small_mask)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
