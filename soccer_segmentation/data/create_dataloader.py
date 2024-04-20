import yaml
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from soccer_segmentation.data.dataloader.dataset import DatasetSegmentation


def get_loader(
        dataset,
        train_idx=None,
        batch_size=32,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
):

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=SubsetRandomSampler(train_idx),
        shuffle=shuffle,
        pin_memory=pin_memory
    )

    return loader


def get_dataset(
        folder_path,
        small_mask=False
):

    dataset = DatasetSegmentation(
        folder_path=folder_path,
        small_mask=small_mask,
    )

    return dataset



if __name__ == "__main__":
    with open("../config.yml") as config_file:
        config = yaml.safe_load(config_file)

    testTransform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), ]
    )

    testLoader, testDataset = get_loader(
        folder_path=config["dataset_path"]["train"],
    )

    for idx, (images, captions) in enumerate(testLoader):
        print(idx)
