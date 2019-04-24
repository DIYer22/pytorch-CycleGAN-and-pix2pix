#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Mon Apr  1 22:18:48 2019

Synthesis glass dataset
"""
from boxx import *
import cv2
from affineFit import affine_fit
from boxx import np, pathjoin, pd, randchoice, imread, os, randfloat, makedirs
import bpp

cf.cloud = sysi.user == 'yanglei'
cf.debug = not cf.cloud

root = os.path.expanduser('~/dataset/celeba')

partition = pd.read_csv(pathjoin(root, 'list_eval_partition.csv'))
trainKeys = partition[partition.partition==0].image_id


imgdir = pathjoin(root, 'img_align_celeba')

attr = 'Eyeglasses'
npyp = pathjoin(pathjoin(root, "%s_cycle_dataset"%attr.lower()), 'fgs.npy')

datasetdir = pathjoin(root, "%s_stgan_dataset"%attr.lower())

trainADir = pathjoin(datasetdir, 'trainA')
makedirs(trainADir)
trainBDir = pathjoin(datasetdir, 'trainB')
makedirs(trainBDir)

glasss = np.load(npyp)

csv = pd.read_csv(pathjoin(root, 'list_attr_celeba.csv'))
notWearKeys = csv[csv[attr]<0].image_id

landmark = pd.read_csv(pathjoin(root, 'list_landmarks_align_celeba.csv'))
landmark = landmark.set_index('image_id')
landmark['image_id'] = landmark.index

fromPoints = (42, 71), (144-42, 71),(42, 71+(144-2*42))

def disturbanceImg(img, maxDu=20, maxScale=0, maxBias=.15):
    rows, cols = img.shape[:2]
    dst = img
    Mrotat = M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (randfloat()-.5)*2*maxDu, 1+(randfloat()-.5)*2*maxScale) 
    dst = cv2.warpAffine(dst, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE) 
    
    M = np.float32([[1, 0, (randfloat()-.5)*2*maxBias*cols], [0, 1, (randfloat()-.5)*2*maxBias*rows]])
    dst = cv2.warpAffine(dst, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE) 
    
    M = mDot(M, Mrotat, )
    return dst, M
def drawAttri(img, d, r = 6, pointName = 'nose'):
    img[d[pointName+'_y']:d[pointName+'_y']+r, d[pointName+'_x']:d[pointName+'_x']+r] = 255
    show-img

def drawPoint(img, point, r=6):
    img[sliceInt[point[1]:point[1]+r, point[0]:point[0]+r]] = (255,0,0)

def mDot(m1, m2):
    if m1.shape != (3,3):
        m1 = np.append(m1, [[0,0,1]], 0)
    if m2.shape != (3,3):
        m2 = np.append(m2, [[0,0,1]], 0)
    return m1.dot(m2)[:2]

keys = set(notWearKeys).intersection(set(trainKeys))

keys = bpp.replicaSplitKeys(keys)

for ind, imgn in enumerate(keys):
    
    trainAP = pathjoin(trainADir, imgn)
    trainBP = pathjoin(trainBDir, imgn)
    try:
        imread(trainAP)
        imread(trainBP)
        continue
    except:
        pass
    
    d = landmark.loc[imgn]
    imgp = pathjoin(imgdir, d.image_id)
    img = imread(imgp)
    
    toPoints = (d.lefteye_x, d.lefteye_y), (d.righteye_x, d.righteye_y), (d.nose_x, d.nose_y)
    toPoints = (d.lefteye_x, d.lefteye_y), (d.righteye_x, d.righteye_y), (d.lefteye_x+d.righteye_y-d.lefteye_y, d.lefteye_y+d.righteye_x-d.lefteye_x)
    M = affine_fit(fromPoints, toPoints).getM()
    rows, cols = img.shape[:2]
    
    glass = randchoice(glasss)
    glass = glasss[-2]
    newGlass = cv2.warpAffine(glass, M, (cols, rows))
#    show-newGlass
    
    glassMask = newGlass[...,-1]>128
    imgGlassed = img.copy()
    imgGlassed[glassMask] = newGlass[...,:3][glassMask]
    def disturbance(img):
        dst, M1 = disturbanceImg(img,20,0.1,0)
#        dst, M1 = disturbanceImg(img,15,0,0.1)
        
        n=1000000
        dx, dy = cols//n,  rows//n
        
        M = np.float32([[1,0,-dx//2],[0,1,-dy//2]])
        dst = cv2.warpAffine(dst, M, (cols-dx, rows-dy)) 
        return dst, mDot(M, M1)

    imgGlassed, M = disturbance(imgGlassed)
#    img = disturbance(img)
    
    gtPoints = [M.dot(p+(1,)) for p in toPoints]
#    for po in gtPoints[:2]:
#        drawPoint(imgGlassed, po)
    
    imsave(trainAP, img)
    imsave(trainBP, imgGlassed)
    
    def avgPooling(img, n=2):
        l = []
        for i in range(n):
            for j in range(n):
#                print(sliceInt[i:img.shape[0]//n*n:n][j:img.shape[1]//n*n:n])
                l.append(1.*img[i:img.shape[0]//n*n:n, j:img.shape[1]//n*n:n])
        mean = sum(l)/len(l)
        mean = mean.round(0).astype(img.dtype)
        import boxx.g
        return mean
    if ind>0 and sysi.user=='dl':
        show-[img,imgGlassed]
        n=32
        show-[avgPooling(img, n),avgPooling(imgGlassed, n)]
        break

#show-dst


if __name__ == "__main__":
    pass
    
    
    
