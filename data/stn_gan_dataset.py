#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Sat Mar 16 15:49:41 2019
"""
from boxx.ylth import *
from .unaligned_dataset import UnalignedDataset
from PIL import Image
from boxx import np, pathjoin, map2, randint

import torch
import torchvision.transforms as T

class StnGanDataset(UnalignedDataset):
    def __init__(self, opt):
        UnalignedDataset.__init__(self, opt)
        
        self.npy = npy = np.load(pathjoin(opt.dataroot, 'fgs.npy'))
#        npy[...,:3][npy[..., -1]<128] = 128 
        
        transform = T.Compose([T.Resize((opt.crop_size, opt.crop_size))]+ self.transform_A.transforms[-2:])
        
        pils = map2(Image.fromarray, npy)
        self.fgs = map2(transform, pils)
        
        self.fg_size = len(self.fgs)
        
#        npy = npy.transpose(0, 3, 1, 2)/255.
        
#        std, mean = .5, .5
        
#        for fg in self.fgs:
#            fg[...,-1,:,:] *= std
#            fg[...,-1,:,:] += mean
            
        
    def __getitem__(self, index):
        d = UnalignedDataset.__getitem__(self, index)
#        d['fg'] = self.fgs[index%self.fg_size]
        d['fg_index'] = randint(self.fg_size-1)
        d['fg'] = self.fgs[d['fg_index']]
        return d
        

if __name__ == "__main__":
    pass
    
    
    
