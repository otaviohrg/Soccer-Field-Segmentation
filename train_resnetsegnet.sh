#!/bin/bash
python -m soccer_segmentation -e resnet18 -d segnet
python -m soccer_segmentation -e resnet34 -d segnet
python -m soccer_segmentation -e resnet50 -d segnet
python -m soccer_segmentation -e resnet101 -d segnet
python -m soccer_segmentation -e resnet152 -d segnet
