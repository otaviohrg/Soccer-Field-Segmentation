#!/bin/bash
set -e

CONFIG="${1:-config.yml}"

ENCODERS_UNET="resnet18 resnet34 resnet50 resnet101 resnet152 vgg11 vgg13 vgg16 vgg19 mobilenetv3small mobilenetv3large default"
ENCODERS_SEGNET="resnet18 resnet34 resnet50 resnet101 resnet152 vgg11 vgg13 vgg16 vgg19 mobilenetv3small mobilenetv3large default"

for enc in $ENCODERS_UNET; do
    echo "========================================"
    echo "Training: encoder=$enc decoder=unet"
    echo "========================================"
    uv run python -m soccer_segmentation train -e "$enc" -d unet --config "$CONFIG"
done

for enc in $ENCODERS_SEGNET; do
    echo "========================================"
    echo "Training: encoder=$enc decoder=segnet"
    echo "========================================"
    uv run python -m soccer_segmentation train -e "$enc" -d segnet --config "$CONFIG"
done
