import numpy as np
import yaml
import argparse
from tqdm import tqdm
from time import sleep

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter

from torchmetrics.classification import Accuracy
from torchmetrics.functional import dice

from soccer_segmentation.data.create_dataloader import get_loader, get_dataset
from soccer_segmentation.utils.checkpoint import load_checkpoint, save_checkpoint
from soccer_segmentation.utils.early_stopping import EarlyStopper
from soccer_segmentation.create_model import create_model


def train_loop(model,
               optimizer,
               criterion,
               step,
               train_loader,
               val_loader,
               device,
               num_epochs,
               use_early_stop=True,
               patience=10,
               min_delta=0,
               starting_epoch=0,
               save_model=True):
    if use_early_stop and patience > 0:
        es_handler = EarlyStopper(patience=patience, min_delta=min_delta)

    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, model.name + ".pth.tar")

        with tqdm(train_loader, unit="batch") as tepoch:
            train_loss = 0.
            train_dice = 0.
            train_line_dice = 0.
            count = 0
            model.train(True)

            for images, masks in tepoch:
                tepoch.update(1)
                tepoch.set_description("Epoch {} - Training".format(epoch + starting_epoch + 1))

                images = images.to(device)
                masks = masks.type(torch.LongTensor).to(device)

                outputs = model(images)
                predictions = torch.from_numpy(np.argmax(outputs.cpu().data.numpy(), 1)).to(device)
                loss = criterion(outputs, masks.squeeze())
                train_loss += loss.item()
                dice_score = dice(predictions, masks, num_classes=3, average='weighted')
                line_dice_score = dice(predictions, masks, num_classes=3, average='none').cpu().numpy()[2]
                count += 1
                train_dice += dice_score
                train_line_dice += line_dice_score
            #
            #     writer.add_scalar("Training loss", loss.item(), global_step=step)
            #     step += 1
            #
                optimizer.zero_grad()
                loss.backward(loss)
                optimizer.step()

                if count == len(tepoch):
                    tepoch.set_postfix({"Train Loss": train_loss/count,
                                        "Train Dice": f"{train_dice/count:.6f}",
                                        "Train Line Dice": f"{train_line_dice/count:.6f}"},
                                       )
                else:
                    tepoch.set_postfix({"Train Loss": loss.item(), "Train Dice": f"{dice_score:.6f}", "Train Line Dice": f"{line_dice_score:.6f}"})
                sleep(0.001)

        with tqdm(val_loader, unit="batch") as tepoch:
            val_loss = 0.
            val_dice = 0.
            val_line_dice = 0.
            count = 0
            model.eval()

            with torch.no_grad():
                for images, masks in tepoch:
                    tepoch.update(1)
                    tepoch.set_description("Epoch {} - Validation".format(epoch + starting_epoch + 1))

                    images = images.to(device)
                    masks = masks.type(torch.LongTensor).to(device)

                    outputs = model(images)
                    predictions = torch.from_numpy(np.argmax(outputs.cpu().data.numpy(), 1)).to(device)
                    loss = criterion(outputs, masks.squeeze())
                    val_loss += loss.item()
                    count += 1
                    dice_score = dice(predictions, masks, num_classes=3, average='weighted')
                    line_dice_score = dice(predictions, masks, num_classes=3, average='none').cpu().numpy()[2]

                    val_dice += dice_score
                    val_line_dice += line_dice_score

                    tepoch.set_postfix({"Validation Loss": val_loss/count, "Val Dice": f"{val_dice/count:.6f}", "Val Line Dice": f"{val_line_dice/count:.6f}"})
                    sleep(0.001)

        if 'es_handler' in locals():
            if es_handler.early_stop(val_loss):
                break


def train():
    parser = argparse.ArgumentParser(
        prog='Soccer Segmentation Trainer',
        description='Trains a segmentation model for robot soccer')

    parser.add_argument('-e', '--encoder')
    parser.add_argument('-d', '--decoder')

    args = parser.parse_args()

    encoder = args.encoder
    decoder = args.decoder

    with open("config.yml") as config_file:
        config = yaml.safe_load(config_file)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    small_mask = True

    # Hyperparameters
    num_classes = 3
    learning_rate = 3e-4
    num_epochs_frozen = 10
    num_epochs_unfrozen = 1000
    batch_size = 32

    # Create Datasets
    train_dataset = get_dataset(
        folder_path=config["dataset_path"]["train"],
        small_mask=small_mask,
    )

    # Tensorboard
#    writer = SummaryWriter()
    train_dataset, val_dataset = data.random_split(train_dataset, [0.7, 0.3])

    train_loader = get_loader(train_dataset.dataset, train_idx=train_dataset.indices, batch_size=batch_size)
    val_loader = get_loader(val_dataset.dataset, train_idx=val_dataset.indices, batch_size=batch_size)

    step = 0

    # Initialize
    model = create_model(encoder, decoder, num_classes, train_encoder=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(model.name + ".pth.tar", model, optimizer)

    print(f"Model: {model.name}")
    print("Starting Transfer Learning Step...")
    train_loop(model, optimizer, criterion, step, train_loader, val_loader, device, num_epochs_frozen,
               starting_epoch=0, save_model=save_model)
    print("Unfreezing Encoder...")
    model.unfreeze()
    print("Continuing Model Training...")
    train_loop(model, optimizer, criterion, step, train_loader, val_loader, device, num_epochs_unfrozen,
               starting_epoch=num_epochs_frozen, save_model=save_model)


if __name__ == "__main__":
    train()
