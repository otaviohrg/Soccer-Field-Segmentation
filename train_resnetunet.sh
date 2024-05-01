#!/bin/bash
python -m soccer_segmentation -e resnet18 -d unet
python -m soccer_segmentation -e resnet34 -d unet
python -m soccer_segmentation -e resnet50 -d unet
