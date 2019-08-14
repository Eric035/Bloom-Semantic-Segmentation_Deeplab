'''
/* Copyright (C) Normative Inc. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Eric Leung <ericleung@normative.com>, August 2019
 */
'''
import cv2
import os, sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

bloom_dir_path_in = "./bloom_dataset/SegmentationClass/"
bloom_dir_path_out = "./bloom_dataset/SegmentationClassRaw/"
pascal_raw_dir_path = "../pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw/"
pascal_jpg_dir_path = "../pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/"

def removeDS_StoreFile(directoryList):          # A function that filters out the '.DS_Store' file
    return list(filter(lambda img_file : '.DS_Store' not in img_file, directoryList))

def drawBorder(img_name, img_path):
    print('Loading image...')
    img_seg = cv2.imread(img_path, 0)
    
    # Dilation
    print('Dilating image...')
    kernel = np.ones((5, 5), np.float64)
    dilation = cv2.dilate(img_seg, kernel, iterations=3)

    # Edge detection
    print('Applying Laplace operator to image...')
    laplacian = cv2.Laplacian(dilation, cv2.CV_64F, ksize=5)

    # Take absolute value of Laplacian
    print('Taking absolute value...')
    abs_laplacian64f = np.absolute(laplacian)

    # Convert abs_lap to int
    laplacian_8u = np.uint8(abs_laplacian64f)

    # Thresholding:
    print('Thresholding...')
    laplacian_8u[(laplacian_8u > 15)] = 255
    laplacian_8u[(laplacian_8u <= 15)] = 0

    mask = laplacian_8u
    print('------------------------------------------------------------------------------------------------')

    dilation[dilation == 21] = 0        # 0: Background
    dilation[(dilation == 222)] = 128   # 128: Plant
    dilation[(mask == 255)] = 255       # 255: Border
    res = dilation
    
    print('Saving image...')
    cv2.imwrite(bloom_dir_path_out + img_name, res)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    seg_img_list = removeDS_StoreFile(os.listdir(bloom_dir_path_in))
    list(map(lambda img : drawBorder(img, bloom_dir_path_in + img), seg_img_list))    # Apply the drawBorder function to each element in our input img list

main()


# expanded = img_seg
# expanded[0:115, 212:255] = 1
# # Left border
# expanded[0:115, 208:211] = 255
# # Right border
# expanded[0:115, 256:259] = 255
# cv2.imshow('expanded border',expanded)
