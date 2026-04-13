import argparse

import torch
import torch.nn as nn
import yaml

from soccer_segmentation.create_model import create_model
from soccer_segmentation.data.create_dataloader import get_loader
from soccer_segmentation.train import evaluate, train
from soccer_segmentation.utils.checkpoint import load_checkpoint
from soccer_segmentation.visualize import visualize


def main():
    parser = argparse.ArgumentParser(
        prog="python -m soccer_segmentation",
        description="Soccer field segmentation toolkit",
    )
    subparsers = parser.add_subparsers(dest="mode", metavar="MODE")
    subparsers.required = True

    train_sub = subparsers.add_parser("train", help="Run the training loop")
    train_sub.add_argument("-e", "--encoder", required=True, help="Encoder backbone")
    train_sub.add_argument("-d", "--decoder", required=True, help="Decoder architecture")
    train_sub.add_argument("--config", default="config.yml", help="Path to config YAML")
    train_sub.add_argument("--resume", action="store_true", help="Load checkpoint before training")
    train_sub.add_argument("--eval-only", action="store_true",
                           help="Evaluate on val set without training (requires --resume)")

    test_sub = subparsers.add_parser("test", help="Evaluate a saved model on the test set")
    test_sub.add_argument("-e", "--encoder", required=True, help="Encoder backbone")
    test_sub.add_argument("-d", "--decoder", required=True, help="Decoder architecture")
    test_sub.add_argument("--config", default="config.yml", help="Path to config YAML")

    viz_sub = subparsers.add_parser("visualize", help="Visualize model predictions")
    viz_sub.add_argument("-e", "--encoder", required=True, help="Encoder backbone")
    viz_sub.add_argument("-d", "--decoder", required=True, help="Decoder architecture")
    viz_sub.add_argument("--config", default="config.yml", help="Path to config YAML")
    viz_group = viz_sub.add_mutually_exclusive_group(required=True)
    viz_group.add_argument("--image", metavar="PATH", help="Path to a single image file")
    viz_group.add_argument("--dataset", metavar="DIR",
                           help="Dataset directory with images/ and segmentations/ sub-dirs")
    viz_sub.add_argument("--annotation", metavar="PATH",
                         help="Path to annotation mask PNG (only with --image)")
    viz_sub.add_argument("--n-samples", type=int, default=4, metavar="N",
                         help="Number of random images to visualize from --dataset (default: 4)")
    viz_sub.add_argument("--output-dir", metavar="DIR",
                         help="Save figures to this directory instead of showing them")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.encoder, args.decoder, args.config, resume=args.resume, eval_only=args.eval_only)
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == "visualize":
        visualize(
            encoder=args.encoder,
            decoder=args.decoder,
            config=config,
            image_path=args.image,
            annotation_path=getattr(args, "annotation", None),
            dataset_path=args.dataset,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config.get("num_classes", 3)
    batch_size = config.get("batch_size", 32)

    model = create_model(args.encoder, args.decoder, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    load_checkpoint(config["checkpoint_path"], model.name + ".pth.tar", model, optimizer)

    test_loader = get_loader(config["dataset_path"]["test"], shuffle=False,
                             small_mask=model.small_mask, batch_size=batch_size)

    print(f"Evaluating {model.name} on {len(test_loader.dataset)} samples...")
    loss, avg_dice, line_dice, acc = evaluate(model, test_loader, criterion, device, num_classes)
    print(f"Test Loss={loss:.4f} | Acc={acc:.4f} | Dice={avg_dice:.4f} | Line Dice={line_dice:.4f}")


if __name__ == "__main__":
    main()
