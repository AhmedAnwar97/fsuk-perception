############ USE ONLY IF YOU HAVE ROS ############

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3

##################################################

import os
import cv2
import copy
import numpy as np
import imgaug as ia
import pandas

def parse_annotation():

    colnames = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']
    data = pandas.read_csv('annotation/via_export_csv.csv', names=colnames)

    name      = data.filename.tolist()
    reg_shape = data.region_shape_attributes.tolist()

    all_ann = []
    all_img = []
    Y_train = [None]*14

    cones = 0
    dim = [None]*2

    while cones < len(name)-1:

        try:
            cones += 1
            
            image = cv2.imread("images/"+str(name[cones]))

            if image is None:
                print("File does not exist")

            else:
                all_img.append(image[:])
                for i in range(7):
                    string = reg_shape[cones].split(',')

                    for j in range(2):
                        temp = string[j+1].split(':')

                        if j==1:
                            temp[1] = temp[1].replace("}", "")
                        
                        dim[j] = int(temp[1])
                    
                    x, y = dim[0], dim[1]
                    
                    Y_train[2*i]   = x
                    Y_train[(2*i)+1] = y
                    cones += 1

                cones -=1
                all_ann.append(Y_train[:])
        
        except FileNotFoundError:
            n = name[cones]
            if n != t:
                print("File does not exist")
                t = name[cones]
            continue
   
    return all_img,all_ann

def normalize(X_train):
    new = []
    for i in range(len(X_train)):
        new.append(X_train[i]/255.)
    return new


# x,y=parse_annotation()
# a=normalize(x)
# print(x)