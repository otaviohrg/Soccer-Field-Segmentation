import glob
import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

torchvision.disable_beta_transforms_warning()
from PIL import Image
from torchvision.transforms import v2

from soccer_segmentation.create_model import create_model

# BGR-friendly colors for each class
CLASS_COLORS = np.array([
    [30,  30,  30],   # 0: background — dark grey
    [255, 255, 255],  # 1: lines      — white
    [34,  139, 34],   # 2: field      — forest green
], dtype=np.uint8)

CLASS_NAMES = ["background", "lines", "field"]

_RESIZE = (356, 356)
_CROP   = (224, 224)

_NORMALIZE = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_SPATIAL = v2.Compose([
    v2.Resize(_RESIZE, antialias=True),
    v2.CenterCrop(_CROP),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


def _load_weights(checkpoint_path, filename, model, device):
    path = os.path.join(checkpoint_path, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])


def _preprocess_image(image_path):
    """Returns (display_np float32 HxWx3 in [0,1], model_input tensor 1x3xHxW)."""
    img = Image.open(image_path).convert("RGB")
    img_t = _SPATIAL(img)                        # (3, 224, 224) in [0, 1]
    display = img_t.permute(1, 2, 0).numpy()     # (224, 224, 3)
    model_input = _NORMALIZE(img_t).unsqueeze(0) # (1, 3, 224, 224)
    return display, model_input


def _preprocess_mask(mask_path):
    """Returns class index array (H, W) int64, matching the dataset encoding."""
    mask = Image.open(mask_path).convert("L")
    mask_t = _SPATIAL(mask).squeeze(0).numpy()   # (224, 224) float32 in [0, 1]

    cls = np.zeros_like(mask_t, dtype=np.int64)
    cls[mask_t <= 0.1] = 0   # background (black)
    cls[mask_t >= 0.9] = 2   # field (white)
    cls[(mask_t > 0.1) & (mask_t < 0.9)] = 1  # lines (gray)
    return cls


def _predict(model, model_input, device):
    """Returns class index array (H, W) int64 at 224×224 resolution."""
    model.eval()
    with torch.no_grad():
        logits = model(model_input.to(device))      # (1, C, H, W)
        pred = logits.argmax(dim=1)                 # (1, H, W)
        if model.small_mask:
            pred = F.interpolate(
                pred.unsqueeze(1).float(), size=_CROP, mode="nearest"
            ).squeeze(1).long()
    return pred.squeeze(0).cpu().numpy()


def _colorize(class_mask):
    """(H, W) int → (H, W, 3) uint8 RGB."""
    return CLASS_COLORS[class_mask]


def _overlay(image_float, color_mask, alpha=0.45):
    """Blend float image [0,1] with uint8 color mask → uint8 (H, W, 3)."""
    img_u8 = (image_float * 255).astype(np.uint8)
    blended = (alpha * color_mask + (1 - alpha) * img_u8).astype(np.uint8)
    return blended


def _legend_patches():
    return [
        mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]


def _render(display_img, pred_mask, ann_mask, output_path):
    has_ann = ann_mask is not None
    ncols = 3 if has_ann else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(display_img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(_overlay(display_img, _colorize(pred_mask)))
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if has_ann:
        axes[2].imshow(_overlay(display_img, _colorize(ann_mask)))
        axes[2].set_title("Annotation")
        axes[2].axis("off")

    fig.legend(handles=_legend_patches(), loc="lower center",
               ncol=len(CLASS_NAMES), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize(encoder, decoder, config,
              image_path=None, annotation_path=None,
              dataset_path=None, n_samples=4,
              output_dir=None):
    """
    Visualize model predictions (and optionally ground-truth annotations).

    Either pass `image_path` for a single image, or `dataset_path` to sample
    `n_samples` random images from a dataset directory.

    Parameters
    ----------
    encoder, decoder : str
        Same identifiers used at training time.
    config : dict
        Parsed config YAML (must contain 'checkpoint_path' and optionally 'num_classes').
    image_path : str, optional
        Path to a single image file.
    annotation_path : str, optional
        Path to the corresponding mask PNG (only used with `image_path`).
    dataset_path : str, optional
        Root dataset directory containing `images/` and `segmentations/` sub-dirs.
    n_samples : int
        How many random images to pick when using `dataset_path`.
    output_dir : str, optional
        Directory to save PNG figures.  If None, figures are shown interactively.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config.get("num_classes", 3)

    model = create_model(encoder, decoder, num_classes).to(device)
    _load_weights(config["checkpoint_path"], model.name + ".pth.tar", model, device)

    # Collect (img_path, ann_path_or_None) pairs
    if image_path:
        samples = [(image_path, annotation_path)]
    else:
        if not dataset_path:
            raise ValueError("Provide either image_path or dataset_path.")
        img_files = sorted(glob.glob(os.path.join(dataset_path, "images", "*")))
        if not img_files:
            raise ValueError(f"No images found under {dataset_path}/images/")
        chosen = random.sample(img_files, min(n_samples, len(img_files)))
        samples = []
        for p in chosen:
            stem = os.path.splitext(os.path.basename(p))[0]
            ann = os.path.join(dataset_path, "segmentations", stem + ".png")
            samples.append((p, ann if os.path.isfile(ann) else None))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for img_p, ann_p in samples:
        display_img, model_input = _preprocess_image(img_p)
        pred_mask = _predict(model, model_input, device)
        ann_mask  = _preprocess_mask(ann_p) if ann_p else None

        out_path = None
        if output_dir:
            stem = os.path.splitext(os.path.basename(img_p))[0]
            out_path = os.path.join(output_dir, f"{stem}_viz.png")

        _render(display_img, pred_mask, ann_mask, out_path)
