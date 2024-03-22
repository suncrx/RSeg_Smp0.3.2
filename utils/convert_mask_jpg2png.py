# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:51:38 2023

@author: renxi
"""

import os, sys
import imutils
import shutil
import numpy as np
import PIL

###############################################################
ROOT_DIR = 'D:/GeoData/DLData/Waters/WaterDataset/val'
IN_FOLDER = 'masks'
OUT_FOLDER = 'masks_2'


out_dir = os.path.join(ROOT_DIR, OUT_FOLDER)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

mask_dir = os.path.join(ROOT_DIR, IN_FOLDER)
maskfiles = os.listdir(mask_dir)
for fn in maskfiles:
    fpath = os.path.join(mask_dir, fn)
    print('Converting '+fpath)
    mask = PIL.Image.open(fpath)
    mask = PIL.ImageOps.grayscale(mask)
    
    m = np.array(mask)    
    m = np.uint8(m>128)*255
            
    msk2 = PIL.Image.fromarray(m)
    
    bfn, ext = os.path.splitext(fn)
    out_path = os.path.join(out_dir, bfn+'.png')
    print(out_path)
    msk2.save(out_path)
    
print('Done!')    