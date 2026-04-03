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
| `unet` | U-Net decoder via `segmentation_models_pytorch` with skip connections |

Pretrained encoders are frozen during the first training phase (transfer learning), then unfrozen for fine-tuning.

The `default` encoder/decoder pair trains a full SegNet or U-Net from scratch with no pretrained weights.

> **Note:** The `unet` decoder is not supported with the `default` encoder — use `segnet` instead.

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

## Dataset download

This project uses the [TORSO-21 dataset](https://github.com/bit-bots/TORSO_21_dataset). To download the real-world split:

```bash
git clone https://github.com/bit-bots/TORSO_21_dataset
cd TORSO_21_dataset
./scripts/download_dataset.py --real
```

Then move the downloaded splits into this project's `data/` folder:

```bash
mv data/reality/train <path-to-this-repo>/data/train
mv data/reality/test  <path-to-this-repo>/data/test
```

The resulting layout expected by the default config is:

```
data/
  train/
    images/
    segmentations/
  test/
    images/
    segmentations/
```

---

## Setup

Copy the example config and fill in your paths:

```bash
cp config.example.yml config.yml
```

```yaml
# config.yml
dataset_path:
  train: data/train
  test:  data/test

checkpoint_path: checkpoints
results_path: results.csv

# Optional — defaults shown
seed: 42
num_classes: 3
learning_rate: 3e-4
batch_size: 32
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
python -m soccer_segmentation train -e default -d segnet
```

Training runs in two phases:

1. **Frozen encoder** — only the decoder is trained (`num_epochs_frozen` epochs)
2. **Unfrozen encoder** — the full network is fine-tuned (`num_epochs_unfrozen` epochs)

A checkpoint is saved to `checkpoint_path/<model_name>.pth.tar` whenever validation loss improves.

Best epoch metrics are appended to `results_path` as a CSV row.

### Resume training

```bash
python -m soccer_segmentation train -e resnet18 -d unet --resume
```

Loads the saved checkpoint and continues training from the loaded weights.

### Evaluate on the validation split

```bash
python -m soccer_segmentation train -e resnet18 -d unet --eval-only
```

Loads the checkpoint and runs one evaluation pass on the validation split. No training is performed.

### Evaluate on the test set

```bash
python -m soccer_segmentation test -e resnet18 -d unet
```

Loads the checkpoint and evaluates on the dataset under `dataset_path.test`.

### Custom config path

Both `train` and `test` accept `--config` to point at a non-default config file:

```bash
python -m soccer_segmentation train -e resnet18 -d unet --config my_config.yml
```

### Batch training

```bash
bash scripts/train_all.sh   # all encoder/decoder combinations
```

---

## Metrics

Each epoch reports loss, accuracy, weighted Dice score, and per-class Dice for the last class (lines), on both the training and validation splits:

```
Epoch 1: Train Loss=0.4821 Acc=0.8123 Dice=0.7103 Line=0.3241 | Val Loss=0.5012 Acc=0.7934 Dice=0.6894 Line=0.2987
```

An `*` is appended when validation loss improves and the checkpoint is saved.

Training stops early if validation loss does not improve for `patience` consecutive epochs.

---

## Project structure

```
soccer_segmentation/
  __main__.py              Entry point (train / test subcommands)
  create_model.py          Model factory — maps encoder+decoder names to instances
  supported_models.py      Lists of valid encoder and decoder names
  train.py                 Training loop, evaluation loop, results logging
  data/
    create_dataloader.py   Dataset loading and train/val splitting
    dataloader/
      dataset.py           PyTorch Dataset — image loading and augmentation
  models/
    encoder_decoder.py     EncoderDecoderModel and SMPModel wrappers
    DefaultSegNet.py       Standalone SegNet (no pretrained encoder)
    DefaultUNet.py         Standalone U-Net (no pretrained encoder)
    encoder/
      resnet.py            ResNet encoder wrapper
      vgg.py               VGG encoder wrapper
      mobilenet.py         MobileNetV3 encoder wrapper
    decoder/
      segnet.py            Parameterised SegNet decoder
  utils/
    checkpoint.py          Save and load training checkpoints
    early_stopping.py      Early stopping based on validation loss
```
