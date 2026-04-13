from definitions import ROOT_DIR
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import random

# Find all the images in the provided images folder
mypath = ROOT_DIR + "/data/test/images"
mask_path = ROOT_DIR + "/data/test/segmentations"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = np.empty(len(onlyfiles), dtype=object)

# Iterate through every image
# and resize all the images.
for n in range(0, len(onlyfiles)):
    mask_file = ".".join(onlyfiles[n].split(".")[:-1]) + ".png"
    path_image = join(mypath, onlyfiles[n])
    path_mask = join(mask_path, mask_file)

    # Load the image in img variable
    img = cv2.imread(path_image)
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)

    # Define a resizing Scale
    # To declare how much to resize
    resize_width = 224
    resize_height = 224
    resized_dimensions = (resize_width, resize_height)

    # Create resized image using the calculated dimensions
    resized_image = cv2.resize(img, resized_dimensions,
                               interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, resized_dimensions,
                               interpolation=cv2.INTER_AREA)

    resized_mask[resized_mask <= 1] = 1
    resized_mask[resized_mask >= 254] = 0
    resized_mask[resized_mask > 1] = 2

    #classes = []
    #for i in range(3):
    #    annotation = resized_mask
    #    annotation[annotation != i] = 255
    #    annotation[annotation == i] = 1
    #    annotation[annotation == 255] = 0
    #    classes.append(annotation)
    #classes = np.array(classes)
    #mask = cv2.merge(classes)


    norm_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    random.seed(path_image)
    factor = random.uniform(0, 1)

    #if factor > 0.75:
    #    suffix = "val"
    #else:
    #    suffix = "train"
    suffix = "test"

    # Save the image in Output Folder
    cv2.imwrite(
        ROOT_DIR + '/data/processed_data/'+suffix+'_images/' + onlyfiles[n], norm_image)
    cv2.imwrite(
        ROOT_DIR + '/data/processed_data/'+suffix+'_segmentation/' + mask_file, resized_mask)
    print(onlyfiles[n])

print("Images resized Successfully")
