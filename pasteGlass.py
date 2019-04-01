#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Mon Apr  1 22:18:48 2019
"""
from boxx import *
import cv2
from affineFit import affine_fit
from boxx import np, pathjoin, pd, randchoice, imread, os

root = os.path.expanduser('~/dataset/celeba')
imgdir = pathjoin(root, 'img_align_celeba')

attr = 'Eyeglasses'
datasetdir = pathjoin(root, "%s_cycle_dataset"%attr.lower())
nSplit2test = 10

npyp = pathjoin(datasetdir, 'fgs.npy')

glasss = np.load(npyp)

csv = pd.read_csv(pathjoin(root, 'list_attr_celeba.csv'))

landmark = pd.read_csv(pathjoin(root, 'list_landmarks_align_celeba.csv'))

fromPoints = (42, 71), (144-42, 71),(42, 71+(144-2*42))


for ind, d in landmark.iterrows():
    imgp = pathjoin(imgdir, d.image_id)
    img = imread(imgp)
    r = 6
    pointName = 'nose'
    img[d[pointName+'_y']:d[pointName+'_y']+r, d[pointName+'_x']:d[pointName+'_x']+r] = 255
#    show-img
    toPoints = (d.lefteye_x, d.lefteye_y), (d.righteye_x, d.righteye_y), (d.nose_x, d.nose_y)
    toPoints = (d.lefteye_x, d.lefteye_y), (d.righteye_x, d.righteye_y), (d.lefteye_x+d.righteye_y-d.lefteye_y, d.lefteye_y+d.righteye_x-d.lefteye_x)
    M = affine_fit(fromPoints, toPoints).getM()
    rows, cols = img.shape[:2]
    
    glass = randchoice(glasss)
    newGlass = cv2.warpAffine(glass, M, (cols, rows))
#    show-newGlass
    
    indMa = newGlass[...,-1]>128
    img[indMa] = newGlass[...,:3][indMa]

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (randfloat()-.5)*40, 1) 
    dst = cv2.warpAffine(img, M, (cols, rows)) 
    
    bias = (np.random.random((2,3))-.5)/3
    bias[0,-1] *= cols
    bias[1,-1] *= rows
    bias[:,:2] = 0 
    
    M = np.float32([[1, 0, 0], [0, 1, 0]]) + bias  
    dst = cv2.warpAffine(dst, M, (cols, rows)) 
    show-dst
#    show-img
    if ind>9:
        break

#show-dst


if __name__ == "__main__":
    pass
    
    
    
