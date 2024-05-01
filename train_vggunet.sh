#!/bin/bash
python -m soccer_segmentation -e vgg11 -d unet
python -m soccer_segmentation -e vgg13 -d unet
python -m soccer_segmentation -e vgg16 -d unet
python -m soccer_segmentation -e vgg19 -d unet
