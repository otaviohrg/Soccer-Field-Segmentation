#!/bin/bash
python -m soccer_segmentation -e resnet101 -d unet
python -m soccer_segmentation -e resnet152 -d unet
