#!/bin/bash
python -m soccer_segmentation -e vgg11 -d segnet
python -m soccer_segmentation -e vgg13 -d segnet
python -m soccer_segmentation -e vgg16 -d segnet
python -m soccer_segmentation -e vgg19 -d segnet
