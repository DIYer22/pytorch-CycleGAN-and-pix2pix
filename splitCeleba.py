#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Sat Mar 16 14:23:43 2019
"""
from boxx import *



root = os.path.expanduser('~/dataset/celeba')
attr = 'Eyeglasses'
nSplit2test = 10



csv = pd.read_csv(pathjoin(root, 'list_attr_celeba.csv'))

unwearInd = csv[attr]<0
dataa =  csv[unwearInd].image_id
datab =  csv[~unwearInd].image_id

testa = dataa[nSplit2test-1::nSplit2test]
traina = [n for n in dataa if n not in testa]


testb = datab[nSplit2test-1::nSplit2test]
trainb = [n for n in datab if n not in testb]


datasetdir = pathjoin(root, "%s_cycle_dataset"%attr.lower())


for sett in ['testA', 'testB', 'trainA', 'trainB']:
    setdir = pathjoin(datasetdir, sett)
    makedirs(setdir)
    for n in locals()[sett.lower()]:
        imgp = pathjoin(root, 'img_align_celeba', n)
        cmd = f'ln {imgp} {setdir}'
        target = pathjoin(setdir, n)
        os.link(imgp, target)
        
#        os.system(cmd)
#        print(cmd)
#        break
if __name__ == "__main__":
    pass
    
    
    
