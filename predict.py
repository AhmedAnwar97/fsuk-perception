#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
import cv2
from utils import draw_boxes
from frontend import keypoints
import json
from keras.preprocessing import image
import scipy.misc
from keras.applications.imagenet_utils import preprocess_input

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    kp = keypoints(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'])

    ###############################
    #   Load trained weights
    ###############################    

    kp.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################
  
    image = cv2.imread(image_path)
    boxes = kp.predict(image)
    # image = draw_boxes(image, boxes, config['model']['labels'])

    print(len(boxes)*7, 'keypoints are found')
    print(boxes)

    # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
