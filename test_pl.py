# %% import packages

import os
import sys

import yaml

import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import SegDataset
from model import SegModel


# %% set parameters

CFG_PATH = 'configs/buildings.yaml'

with open(CFG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
print('Config: ')
print(cfg)


# %% load model
# generate model file path from parameters.
mname = cfg['test_mfile'].replace(' ', '')
if mname == '':
    mname = 'nomodel'    
mdl_path = os.path.join(cfg['root_dir'], mname.lower())
if not os.path.exists(mdl_path):
    #!!! model name format 
    default_mdl_name = 'smp_%s_%s.pth' % (cfg['marct'], cfg['mencoder'])
    mdl_path = os.path.join(cfg['root_dir'], default_mdl_name.lower())
    
if not os.path.exists(mdl_path):
    print('Model path does not exist: ', mdl_path)
    sys.exit()

# is binary segmentation ?
binary_seg = cfg['mclasses'] <= 2

print('Loading model ...')
mod = SegModel(cfg['marct'], 
               cfg['mencoder'], 
               in_channels=cfg['mchannels'], 
               out_classes=cfg['mclasses'])
mod.model.load_state_dict(torch.load(mdl_path))
print('[INFO]: Model loaded: %s\n' % mdl_path)



# %% create model and load check point from check-point file
'''
chkpnt_path = os.path.join(cfg.DATA_ROOT, CKF)
if not os.path.exists(chkpnt_path):
    print('Check point path does not exist: ', chkpnt_path)
    sys.exit()

print('Loading checkpoint: ', chkpnt_path)
mod = PetModel.load_from_checkpoint(chkpnt_path, 
                                    cfg.M_ARCT, 
                                    encoder_name = cfg.M_ENCODER, 
                                    in_channels=cfg.M_IN_CHANNELS, 
                                    out_classes=cfg.M_OUT_CLASSES)
'''


# %% create test date loader
test_dataset = SegDataset(cfg['root_dir'], "test")
print(f"Test data size: {len(test_dataset)}")
test_dataloader = DataLoader(test_dataset, batch_size=cfg['test_btsize'], 
                             shuffle=False, num_workers=0)


# %% run test dataset
print('\nTesting model ...')
trainer = pl.Trainer(accelerator="auto", devices="auto")
# all images in test_data will be fed  to the trained network and the evaluation
# metric will be returned.
test_metrics = trainer.test(mod, dataloaders=test_dataloader, verbose=True)
print('Testing metric: ')
print(test_metrics)


# %% visualize results
# create output directory
out_dir = os.path.join(cfg['root_dir'], cfg['out_folder'], 
                       cfg['marct']+'_'+cfg['mencoder'])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# iterate images in datasets
print('Generating results ... ')
nImgs = 0
for i, batch in enumerate(test_dataloader):
    print("batch: ",  i+1)
    with torch.no_grad():
        mod.eval()
        # get results from the network
        out = mod(batch["image"])
    
    # binary segmentation
    if binary_seg: 
        # create prob-mask by applying sigmoid on the results
        # the values in pr_mask range in (0 ~ 1.0)
        #pr_masks = out.sigmoid()    
        pr_masks = out
    # multi-class segmentation        
    else:
        # determine the class by the index with the maximum along dimension C
        # out format : [batches, C, H, W]
        # ??? to be completed ... 
        pr_masks = np.uint8(torch.argmax(out, dim=1))
    
    
    for image, gt_mask, pr_mask in zip(batch["oimage"], batch["omask"], pr_masks):
        #save predicted mask image-------------------------------
        maskimg = pr_mask.numpy().squeeze()
        # binary segmentation 
        if binary_seg:
            maskimg = np.uint8(maskimg*255)        
        
        #maskimg = Image.fromarray(maskimg)
        #maskimg.save(os.path.join(out_dir, '%d.png' % nImgs))
        cv2.imwrite(os.path.join(out_dir, '%d.png' % nImgs), maskimg)
        
        # NOTE: oimage and omask from batch['oimage'], batch['omask'] have been
        # transformed into pytorch tensor. You need to transfrom them back into
        # numpy format before saving them.
        #save ground-truth mask image
        cv2.imwrite(os.path.join(out_dir, '%d_gt.png' % nImgs), gt_mask.numpy())
        #save rgb image
        cv2.imwrite(os.path.join(out_dir, '%d.jpg' % nImgs), image.numpy())
        
        nImgs = nImgs + 1
        
        
        #visualizing the results----------------------------------
        if not cfg['display']:
            continue
        
        plt.figure(figsize=(10, 5))
    
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy())  
        plt.title("Image")
        plt.axis("off")
    
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy())  
        plt.title("Ground truth")
        plt.axis("off")
    
        plt.subplot(1, 3, 3)
        plt.imshow(maskimg)  
        plt.title("Prediction")
        plt.axis("off")
    
        plt.show()

print('Finished')
print('See results at %s' % out_dir)