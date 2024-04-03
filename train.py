# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:38:14 2023

@author: renxi
"""

#!pip install segmentation-models-pytorch
#!pip install pytorch-lightning==1.5.4
# Ref: https://lightning.ai/docs/pytorch/1.5.4/

#usage example:
# python train.py --data ./data/waters.yaml out_dir './output' --arct unet --encoder resnet34
# --imgsz 512 --epochs 2 --batch_size 4 --lr 0.001 --momentum 0.9 --checkpoint True 

# %% import installed packages
import os
import sys
import time
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import matplotlib.pylab as plt

import segmentation_models_pytorch as smp

# explictly import utils if segmentation_models_pytorch >= 0.3.2
from segmentation_models_pytorch import utils as smp_utils 

import models
from dataset import SegDataset
from utils.common import log_csv

#%% get current directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # FasterRCNN root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# determine the device to be used for training and evaluation
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print('Device: ', DEV)
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEV == "cuda" else False


#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default=ROOT/'data/vehicles.yaml', 
                        help='dataset yaml path')
    
    parser.add_argument('--img_sz', '--img', '--img-size', type=int, 
                        default=256, help='train, val image size (pixels)')
    
    parser.add_argument('--out_dir', type=str, default='', 
                        help='training output path')    
    
    parser.add_argument('--arct', type=str, default='munet', 
                        help='model architecture (options: unet, unetplusplus, manet, linknet, fpn, pspnet, deeplabv3,deeplabv3plus, pan')

    parser.add_argument('--encoder', type=str, default='resnet34', 
                        help='encoder for the net (options: resnet34, resnet50, vgg16, vgg19')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='momentum')
    #parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
    
    parser.add_argument('--aug', type=bool, default=True, 
                        help='Data augmentation')
    parser.add_argument('--sub_size', type=float, default=1.0, 
                        help='subsize of training data')
    
    parser.add_argument('--checkpoint', type=bool, default=True, 
                        help='enable checking point')
            
    return parser.parse_args()


#%% run training
def run(opt): 
    #%% parameters
    data_yaml_file, img_sz = opt.data, opt.img_sz
    arct, encoder = opt.arct, opt.encoder
    batch_size, epochs = opt.batch_size, opt.epochs
    lr, momentum = opt.lr, opt.momentum
    #check_point = opt.checkpoint
    out_dir = opt.out_dir
    
    sub_size = opt.sub_size
    #applying data augumentation or not
    bAug = opt.aug
    
    encoder_weight = 'imagenet'
    
    # read data information from yaml file          
    assert os.path.exists(data_yaml_file)
    with open(data_yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # cfg is a dictionary
    if 'exp_name' in cfg.keys():
        print(cfg['exp_name'])
    print('Data information:', cfg)
    n_classes = cfg['nclasses']
    n_channels = cfg['nchannels']
    class_names = cfg['names']
    root_data_dir = cfg['path']
    train_folder = cfg['train']
    val_folder = cfg['val']
    if out_dir == '':
        out_dir = os.path.join(root_data_dir, 'out')    
    os.makedirs(out_dir, exist_ok=True)
    
    # %% prepare datasets
    # init train, val, test sets
    print('Preparing data ...')
    
    #preprocessing function from segmentation-models-pytorch package
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)

    train_dataset = SegDataset(root_data_dir, "train", 
                               n_classes=n_classes, imgH=img_sz, imgW=img_sz, 
                               preprocess=preprocessing_fn,
                               apply_aug = bAug, sub_size=sub_size)
    val_dataset = SegDataset(root_data_dir, "val", 
                             n_classes=n_classes, imgH=img_sz, imgW=img_sz,
                             preprocess=preprocessing_fn)
    
    # It is a good practice to check datasets don't intersects with each other
    train_imgs = train_dataset.get_image_filepaths()
    val_imgs = val_dataset.get_image_filepaths()
    assert set(val_imgs).isdisjoint(set(train_imgs))
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(val_dataset)}")
    #print(f"Test size: {len(test_dataset)}")
        
    n_cpu = 0
    #n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=n_cpu)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=n_cpu)
    
    
    #%% initialize model
    model, model_name = models.utils.create_model(arct=arct, 
                                                  encoder=encoder,
                                                  encoder_weigths=encoder_weight,
                                                  n_classes = n_classes,
                                                  in_channels = n_channels)
    if model is None:
        print("ERROR: cannot create a model named '%s'" % model_name)
        sys.exit(0)
    
    # loss function
    #lossFunc = smp.utils.losses.DiceLoss()         #version 0.2.1
    lossFunc = smp_utils.losses.DiceLoss()          #version 0.3.2
    
    # metrics
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]
    
    # optimizer
    opt = optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.999],
                     eps=1e-7, amsgrad=False)
    #opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # learning rate scheduler
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        
    #%% training
    print("\nTraining network: %s ..." % model_name)
    
    train_epoch = smp_utils.train.TrainEpoch(model, loss=lossFunc, 
                                             metrics=metrics,
                                             optimizer=opt, device=DEV, verbose=True)
    
    val_epoch = smp_utils.train.ValidEpoch(model, loss=lossFunc, 
                                           metrics=metrics,
                                           device=DEV, verbose=True )
    
    
    startTime = time.time()
    max_score = -np.inf
    best_epoch = 0
    train_losses, val_losses = [], []
    train_scores, val_scores = [], []
    for i in range(0, epochs):
        time_se = time.time()
        print('\nEpoch: %d/%d' % (i+1, epochs))
        train_logs = train_epoch.run(train_dataloader)
        val_logs = val_epoch.run(val_dataloader)
    	
        train_losses.append(train_logs['dice_loss'])
        val_losses.append(val_logs['dice_loss'])
        
        tsc = train_logs['iou_score']
        vsc = val_logs['iou_score']
        train_scores.append(tsc)	
        val_scores.append(vsc)	
        
        log_csv(train_losses, val_losses, train_scores, val_scores, 
                os.path.join(out_dir, model_name+'_log.csv'))
        endTime = time.time()
        print("Elapsed time : {:.3f}s".format(endTime - time_se))
        if vsc > max_score:
            max_score = vsc        
            best_epoch = i+1
            mdlpath = os.path.join(out_dir, model_name+'_best.pt')        
            models.utils.save_seg_model(model, mdlpath, arct, encoder, 
                                        n_classes, class_names, n_channels)
            print('Best epoch: %d, Best model saved:%s' % (best_epoch, mdlpath))
        
    endTime = time.time()
    print("\nTotal time : {:.2f}s".format(endTime - startTime))
    
    mdlpath = os.path.join(out_dir, model_name+'_last.pt')        
    models.utils.save_seg_model(model, mdlpath, arct, encoder, 
                                n_classes, class_names, n_channels)
    print('Last model: %s' % mdlpath)
        
        
    #%% plot training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="validation_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Dice Loss")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(out_dir, model_name+'_loss.png'), dpi=200)
    
    plt.figure()
    plt.plot(train_scores, label="train_score")
    plt.plot(val_scores, label="validation_score")
    plt.title("Training score on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("IoU")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(out_dir, model_name+'_IoU.png'), dpi=200)

    
    print('Finished')    

#%% calling
if __name__ == '__main__':
    opt = parse_opt()
    #print(opt)
    run(opt)