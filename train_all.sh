#!/bin/bash
python -m soccer_segmentation -e resnet18 -d unet
python -m soccer_segmentation -e resnet34 -d unet
python -m soccer_segmentation -e resnet50 -d unet
python -m soccer_segmentation -e resnet101 -d unet
python -m soccer_segmentation -e resnet152 -d unet
python -m soccer_segmentation -e mobilenetv3small -d unet
python -m soccer_segmentation -e mobilenetv3large -d unet
python -m soccer_segmentation -e resnet18 -d segnet
python -m soccer_segmentation -e resnet34 -d segnet
python -m soccer_segmentation -e resnet50 -d segnet
python -m soccer_segmentation -e resnet101 -d segnet
python -m soccer_segmentation -e resnet152 -d segnet
python -m soccer_segmentation -e mobilenetv3small -d segnet
python -m soccer_segmentation -e mobilenetv3large -d segnet



