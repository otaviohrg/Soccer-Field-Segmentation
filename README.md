# Soccer Field Segmentation

Encoder-decoder CNN for semantic segmentation of soccer fields, trained to distinguish three classes: **background**, **field**, and **lines**.

---

## Architecture

The project follows a modular encoder-decoder design. Any supported encoder can be paired with any supported decoder.

### Encoders

| Name | Backbone | Pretrained |
|---|---|---|
| `resnet18` / `resnet34` / `resnet50` / `resnet101` / `resnet152` | ResNet | ImageNet |
| `vgg11` / `vgg13` / `vgg16` / `vgg19` | VGG | ImageNet |
| `mobilenetv3small` / `mobilenetv3large` | MobileNetV3 | ImageNet |
| `default` | From scratch (VGG-style) | — |

### Decoders

| Name | Description |
|---|---|
| `segnet` | SegNet-style decoder using max-pool indices for upsampling |
| `unet` | U-Net decoder with skip connections and transposed convolutions |

Pretrained encoders are frozen during the first training phase (transfer learning), then unfrozen for fine-tuning.

The `default` encoder/decoder pair trains a full SegNet or U-Net from scratch with no pretrained weights.

---

## Dataset structure

The dataset directory must follow this layout:

```
dataset/
  images/           ← input RGB images
  segmentations/    ← grayscale masks, same basename as the corresponding image
```

Masks use three pixel values to encode classes:

| Pixel value | Class |
|---|---|
| ~0 (black) | Background |
| ~128 (grey) | Lines |
| ~255 (white) | Field |

At load time the dataset rescales to 356×356, applies a random 224×224 crop, normalises with ImageNet statistics, and remaps mask values to integer class indices `{0, 1, 2}`. Encoders that output a spatially reduced feature map receive a 112×112 mask instead (`small_mask` mode).

---

## Setup

Copy the example config and fill in your paths:

```bash
cp config.example.yml config.yml
```

```yaml
# config.yml
dataset_path:
  train: /path/to/train/data
  test:  /path/to/test/data

checkpoint_path: /path/to/save/weights

# Optional — defaults shown
seed: 42
num_classes: 3
learning_rate: 3e-4
batch_size: 32
val_split: 0.3
num_epochs_frozen: 10
num_epochs_unfrozen: 10
patience: 10
```

`seed` controls the train/val split so results are reproducible across runs.

---

## Usage

### Train

```bash
python -m soccer_segmentation train -e <encoder> -d <decoder>
```

Examples:

```bash
python -m soccer_segmentation train -e resnet18 -d unet
python -m soccer_segmentation train -e vgg16 -d segnet
python -m soccer_segmentation train -e default -d unet
```

Training runs in two phases:

1. **Frozen encoder** — only the decoder is trained (`num_epochs_frozen` epochs)
2. **Unfrozen encoder** — the full network is fine-tuned (`num_epochs_unfrozen` epochs)

A checkpoint is saved after each epoch to `checkpoint_path/<model_name>.pth.tar`.

### Resume training

```bash
python -m soccer_segmentation train -e resnet18 -d unet --resume
```

Loads the saved checkpoint and continues training from the loaded weights.

### Evaluate on the validation split

```bash
python -m soccer_segmentation train -e resnet18 -d unet --eval-only
```

Loads the checkpoint and runs one evaluation pass on the validation split. No training is performed. Useful for inspecting a model without touching the held-out test set.

### Evaluate on the test set

```bash
python -m soccer_segmentation test -e resnet18 -d unet
```

Loads the checkpoint and evaluates on the dataset under `dataset_path.test`.

### Batch training

Pre-written shell scripts are provided for common combinations:

```bash
bash train_all.sh           # all encoder/decoder combinations
bash train_resnetunet.sh    # ResNet variants with U-Net
bash train_resnetsegnet.sh  # ResNet variants with SegNet
bash train_vggunet.sh
bash train_vggsegnet.sh
bash train_mobilenet.sh
bash train_default.sh
```

---

## Metrics

Each epoch reports loss, weighted Dice score, and per-class Dice for the last class (lines), on both the training and validation splits:

```
Epoch 1: Train Loss=0.4821 Dice=0.7103 Line=0.3241 | Val Loss=0.5012 Dice=0.6894 Line=0.2987
```

Training stops early if validation loss does not improve for `patience` consecutive epochs.

---

## Project structure

```
soccer_segmentation/
  __main__.py              Entry point (train / test subcommands)
  create_model.py          Model factory — maps encoder+decoder names to instances
  supported_models.py      Lists of valid encoder and decoder names
  train.py                 Training loop, evaluation loop, CLI entry point
  data/
    create_dataloader.py   Dataset loading and train/val splitting
    dataloader/
      dataset.py           PyTorch Dataset — image loading and augmentation
  models/
    encoder_decoder.py     Generic wrapper combining any encoder and decoder
    DefaultSegNet.py       Standalone SegNet (no pretrained encoder)
    DefaultUNet.py         Standalone U-Net (no pretrained encoder)
    encoder/
      resnet.py            ResNet encoder wrapper
      vgg.py               VGG encoder wrapper
      mobilenet.py         MobileNetV3 encoder wrapper
    decoder/
      segnet.py            Parameterised SegNet decoder
      unet_v1.py           U-Net decoder for VGG encoders
      unet_v2.py           U-Net decoder for ResNet18/34
      unet_v3.py           U-Net decoder for ResNet50+
      unet_v4.py           U-Net decoder for MobileNetV3 Small
      unet_v5.py           U-Net decoder for MobileNetV3 Large
  utils/
    checkpoint.py          Save and load training checkpoints
    early_stopping.py      Early stopping based on validation loss
```
