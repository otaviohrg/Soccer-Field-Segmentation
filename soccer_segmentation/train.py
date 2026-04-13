import csv
import os
import time
import yaml
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore

from soccer_segmentation.data.create_dataloader import get_loader, get_train_val_loaders
from soccer_segmentation.utils.checkpoint import load_checkpoint, save_checkpoint
from soccer_segmentation.utils.early_stopping import EarlyStopper
from soccer_segmentation.create_model import create_model


def _batch_metrics(outputs, masks, loss, num_classes):
    predictions = outputs.argmax(dim=1)
    device = predictions.device
    weighted = DiceScore(num_classes=num_classes, average='weighted', input_format='index').to(device)
    per_class = DiceScore(num_classes=num_classes, average='none', input_format='index').to(device)
    accuracy = (predictions == masks).float().mean().item()
    w = weighted(predictions, masks)
    pc = per_class(predictions, masks)
    return (
        loss.item(),
        0.0 if w.isnan() else w.item(),
        0.0 if pc[1].isnan() else pc[1].item(),
        accuracy,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes, desc="Training"):
    model.train()
    total_loss = total_dice = total_line_dice = total_acc = 0.0

    with tqdm(loader, desc=desc, unit="batch", leave=False) as pbar:
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.squeeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss, batch_dice, batch_line_dice, batch_acc = _batch_metrics(outputs, masks, loss, num_classes)
            total_loss += batch_loss
            total_dice += batch_dice
            total_line_dice += batch_line_dice
            total_acc += batch_acc
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n, total_line_dice / n, total_acc / n


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = total_dice = total_line_dice = total_acc = 0.0
    total_time_ms = 0.0
    total_images = 0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        with tqdm(loader, desc="Validation", unit="batch", leave=False) as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.squeeze(1).to(device)

                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = model(images)
                if use_cuda:
                    torch.cuda.synchronize()
                total_time_ms += (time.perf_counter() - t0) * 1000
                total_images += images.size(0)

                loss = criterion(outputs, masks)

                batch_loss, batch_dice, batch_line_dice, batch_acc = _batch_metrics(outputs, masks, loss, num_classes)
                total_loss += batch_loss
                total_dice += batch_dice
                total_line_dice += batch_line_dice
                total_acc += batch_acc
                pbar.set_postfix(loss=f"{batch_loss:.4f}")

    n = len(loader)
    inference_ms = total_time_ms / total_images
    return total_loss / n, total_dice / n, total_line_dice / n, total_acc / n, inference_ms


def train_loop(model, optimizer, criterion, train_loader, val_loader, device,
               num_classes, num_epochs, checkpoint_path, patience=10, starting_epoch=0):
    """
    Runs the training loop for num_epochs, saving a checkpoint only when val_loss improves.
    Returns a list of per-epoch metric dicts.
    """
    es = EarlyStopper(patience=patience) if patience > 0 else None
    best_val_loss = float('inf')
    best_metrics = None
    history = []

    for epoch in range(num_epochs):
        epoch_num = epoch + starting_epoch + 1
        train_loss, train_dice, train_line_dice, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes,
            desc=f"Epoch {epoch_num} - Training",
        )
        val_loss, val_dice, val_line_dice, val_acc, inference_ms = evaluate(
            model, val_loader, criterion, device, num_classes,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "step": epoch_num},
                checkpoint_path,
                model.name + ".pth.tar",
            )

        print(
            f"Epoch {epoch_num}: "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} Dice={train_dice:.4f} Line={train_line_dice:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} Dice={val_dice:.4f} Line={val_line_dice:.4f} InfMs={inference_ms:.2f}"
            + (" *" if improved else "")
        )

        metrics = {
            "epoch": epoch_num,
            "train_loss": train_loss, "train_acc": train_acc, "train_dice": train_dice, "train_line_dice": train_line_dice,
            "val_loss": val_loss, "val_acc": val_acc, "val_dice": val_dice, "val_line_dice": val_line_dice,
            "val_inference_ms": inference_ms,
        }
        history.append(metrics)

        if improved:
            best_metrics = metrics

        if es and es.early_stop(val_loss):
            print("Early stopping triggered.")
            break

    return history, best_metrics


_VAL_CSV_FIELDS = ["model", "epoch",
                   "train_loss", "train_acc", "train_dice", "train_line_dice",
                   "val_loss",   "val_acc",   "val_dice",   "val_line_dice",
                   "val_inference_ms"]

_TEST_CSV_FIELDS = ["model", "test_loss", "test_acc", "test_dice", "test_line_dice", "test_inference_ms"]


def _log_results(results_path, model_name, best_metrics):
    row = {"model": model_name, **{k: f"{best_metrics[k]:.6f}" if isinstance(best_metrics[k], float) else best_metrics[k]
                                   for k in _VAL_CSV_FIELDS[1:]}}
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    write_header = not os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_VAL_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _log_test_results(results_path, model_name, loss, acc, dice, line_dice, inference_ms):
    row = {
        "model": model_name,
        "test_loss": f"{loss:.6f}",
        "test_acc": f"{acc:.6f}",
        "test_dice": f"{dice:.6f}",
        "test_line_dice": f"{line_dice:.6f}",
        "test_inference_ms": f"{inference_ms:.6f}",
    }
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    write_header = not os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TEST_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(encoder, decoder, config_path="config.yml", resume=False, eval_only=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    num_classes = config.get("num_classes", 3)
    learning_rate = float(config.get("learning_rate", 3e-4))
    batch_size = config.get("batch_size", 32)
    num_epochs_frozen = config.get("num_epochs_frozen", 10)
    num_epochs_unfrozen = config.get("num_epochs_unfrozen", 10)
    patience = config.get("patience", 10)
    val_size = config.get("val_size", 1570)
    checkpoint_path = config["checkpoint_path"]
    results_path = config.get("results_path", "results.csv")
    test_results_path = config.get("test_results_path", "test_results.csv")

    model = create_model(encoder, decoder, num_classes, train_encoder=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if resume or eval_only:
        step = load_checkpoint(checkpoint_path, model.name + ".pth.tar", model, optimizer)
        print(f"Loaded checkpoint (step {step})")

    train_loader, val_loader = get_train_val_loaders(
        config["dataset_path"]["train"], val_size=val_size, seed=seed,
        small_mask=model.small_mask, batch_size=batch_size,
    )
    test_loader = get_loader(config["dataset_path"]["test"], shuffle=False,
                             small_mask=model.small_mask, batch_size=batch_size)

    print(f"Model: {model.name} | Device: {device} | "
          f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    if eval_only:
        loss, avg_dice, line_dice, acc, inference_ms = evaluate(model, val_loader, criterion, device, num_classes)
        print(f"Val Loss={loss:.4f} | Acc={acc:.4f} | Dice={avg_dice:.4f} | Line Dice={line_dice:.4f} | InfMs={inference_ms:.2f}")
        return

    print("Starting transfer learning (encoder frozen)...")
    _, best_frozen = train_loop(model, optimizer, criterion, train_loader, val_loader, device,
                                num_classes, num_epochs_frozen, checkpoint_path, patience=patience,
                                starting_epoch=0)

    print("Unfreezing encoder...")
    model.unfreeze()

    print("Continuing training (encoder unfrozen)...")
    _, best_unfrozen = train_loop(model, optimizer, criterion, train_loader, val_loader, device,
                                  num_classes, num_epochs_unfrozen, checkpoint_path, patience=patience,
                                  starting_epoch=num_epochs_frozen)

    best = min((m for m in [best_frozen, best_unfrozen] if m is not None),
               key=lambda m: m["val_loss"])
    print(
        f"\nBest epoch {best['epoch']}: "
        f"Val Loss={best['val_loss']:.4f} Acc={best['val_acc']:.4f} "
        f"Dice={best['val_dice']:.4f} Line={best['val_line_dice']:.4f} InfMs={best['val_inference_ms']:.2f}"
    )
    _log_results(results_path, model.name, best)

    print("Running inference on test set...")
    load_checkpoint(checkpoint_path, model.name + ".pth.tar", model, optimizer)
    test_loss, test_dice, test_line_dice, test_acc, test_inference_ms = evaluate(
        model, test_loader, criterion, device, num_classes,
    )
    print(
        f"Test Loss={test_loss:.4f} Acc={test_acc:.4f} "
        f"Dice={test_dice:.4f} Line={test_line_dice:.4f} InfMs={test_inference_ms:.2f}"
    )
    _log_test_results(test_results_path, model.name, test_loss, test_acc, test_dice, test_line_dice, test_inference_ms)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="soccer_segmentation train",
        description="Train a segmentation model for robot soccer",
    )
    parser.add_argument("-e", "--encoder", required=True, help="Encoder backbone")
    parser.add_argument("-d", "--decoder", required=True, help="Decoder architecture")
    parser.add_argument("--config", default="config.yml", help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Load checkpoint before training")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate on val set, skip training")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    train(args.encoder, args.decoder, args.config, resume=args.resume, eval_only=args.eval_only)
