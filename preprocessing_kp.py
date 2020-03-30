import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, bbox_iou
import pandas

def parse_annotation_kp(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    colnames = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']
    data = pandas.read_csv('annotations/via_export_csv.csv', names=colnames)

    name = data.filename.tolist()
    region_shape = data.region_shape_attributes.tolist()
    
    cones = 0
    dim = [None] * 4
    length = len(name)
    t = name[0]
    image_count = 0

    while cones < len(name)-1:
        try:
            cones += 1
            image = cv2.imread("images/"+str(name[cones]))
            
            if image is None:
                print("File does not exist")

            else:
                print(name[cones])
                string = region_shape[cones].split(',')
                print(string)

                for j in range(4):
                    temp = string[j+1].split(':')

                    if j==3:
                        temp[1] = temp[1].replace("}", "")
                    
                    dim[j] = int(temp[1])
                seen_labels += dim[j]

                all_imgs += [image]
        
        except FileNotFoundError:
            n = name[cones]
            if n != t:
                print("File does not exist")
                t = name[cones]
            continue
                        
    return all_imgs, seen_labels