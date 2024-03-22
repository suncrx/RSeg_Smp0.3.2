# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:38:14 2023

@author: renxi
"""

#!pip install segmentation-models-pytorch
#!pip install pytorch-lightning==1.5.4
# Ref: https://lightning.ai/docs/pytorch/1.5.4/

#usage example:
# python train.py --data ./data/waters.yaml out_dir './output' --net unet --backbone resnet34
# --imgsz 512 --epochs 2 --batch_size 4 --lr 0.001 --momentum 0.9 --checkpoint True 

# %% import installed packages
import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import model 
from dataset import SegDataset
from plot_logs import plot_csv


#%% get current directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # FasterRCNN root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default=ROOT / 'data/buildings.yaml', 
                        help='dataset yaml path')
    
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, 
                        default=512, help='train, val image size (pixels)')
    
    parser.add_argument('--out_dir', type=str, default=ROOT / 'out', 
                        help='training output path')    
    
    parser.add_argument('--net', type=str, default='unet', 
                        help='model name (options: unet, unetplusplus, manet, linknet, fpn, pspnet, deeplabv3,deeplabv3plus, pan')

    parser.add_argument('--backbone', type=str, default='resnet34', 
                        help='backbone net (options: resnet34, resnet50, vgg16, vgg19')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    #parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
    
    parser.add_argument('--checkpoint', type=bool, default=True, help='enable checking point')
    
            
    return parser.parse_args()


#%% run training
def run(opt): 
    #%% parameters
    data_yaml_file, imgsz = opt.data, opt.imgsz
    backbone_name, net_name = opt.backbone, opt.net
    batch_size, epochs = opt.batch_size, opt.epochs
    lr, momentum = opt.lr, opt.momentum
    check_point = opt.checkpoint
    out_dir = opt.out_dir
    
    # read data information from yaml file          
    assert os.path.exists(data_yaml_file)
    with open(data_yaml_file) as f:
        datainfo = yaml.load(f, Loader=yaml.SafeLoader)
    # cfg is a dictionary
    print('data information:', datainfo)
    n_classes = datainfo['nclasses']
    n_channels = datainfo['nchannels']
    class_names = datainfo['names']
    
    root_data_dir = datainfo['path']
    #val_img_dir = datainfo['val']
    #if 'path' in datainfo.keys:
    #    train_img_dir = os.path.join(datainfo['path'], train_img_dir)        
    #    val_img_dir = os.path.join(datainfo['path'], val_img_dir)
    
    #applying data augumentation or not
    bAug = False
        
    # %% prepare datasets
    # init train, val, test sets
    print('Prepare data ...')
    train_dataset = SegDataset(root_data_dir, "train", 
                               n_classes=1, imgH=imgsz, imgW=imgsz, 
                               apply_aug = bAug) #, sub_size=data_subsize)
    valid_dataset = SegDataset(root_data_dir, "val", 
                               n_classes=1, imgH=imgsz, imgW=imgsz)
    
    # It is a good practice to check datasets don't intersects with each other
    train_imgs = train_dataset.get_image_filepaths()
    val_imgs = valid_dataset.get_image_filepaths()
    #test_imgs = test_dataset.get_image_filepaths()
    #assert set(test_imgs).isdisjoint(set(train_imgs))
    #assert set(test_imgs).isdisjoint(set(val_imgs))
    assert set(val_imgs).isdisjoint(set(train_imgs))
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    #print(f"Test size: {len(test_dataset)}")
    
    
    n_cpu = 0
    #n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=n_cpu)
    
    
    #------------------------------------------------------------------------------
    # %% create model
    # you can change model in configuration file
    print('Start training ...')
    lpmod = model.SegModel(net_name, backbone_name, 
                         in_channels=n_channels, 
                         out_classes=n_classes)
    
    
    #%% create csv logger
    log_dir = os.path.join(root_data_dir, 'logs')
    log_name = net_name + '_' + backbone_name
    csvlog = pl.loggers.CSVLogger(log_dir, name=log_name)
    
    
    #%% check point callback
    # moniter is the metric that is used to evaluate the model.
    # see the following function of class SegModel in model.py
    #    def shared_epoch_end(self, outputs, stage):
    #        ....
    # By default, the check-point model will be saved  to 
    # the location specified by Trainer’s default_root_dir argument, 
    # and if the Trainer uses a logger, the path will also contain 
    # logger name and version.
    ckp_cb = ModelCheckpoint(monitor='valid_dataset_iou',
                             mode = 'max',
                             save_top_k = 1,
                             save_weights_only=False)
    
    #%% training
    # setting full batches or partial batches [default values = 1.0]
    # float number 1.0 means training is performed on full data batches
    # if limit_batch is True, the max train_batch is 20, and the max
    # val_batch is 5. 
    train_batches = 1.0
    val_batches = 1.0
#    if cfg['limit_batch']:
#        print('Training is under limit batchs.')
#        train_batches = 20
#        val_batches = 5
    
    trainer = pl.Trainer(max_epochs=epochs, 
                         default_root_dir=root_data_dir,                     
                         accelerator="auto",
                         devices="auto",
                         enable_checkpointing=check_point,
                         logger = csvlog, 
                         callbacks = [ckp_cb],
                         limit_train_batches=train_batches,
                         limit_val_batches=val_batches)
    
    '''
    Trainer.fit(model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None)
    ckpt_path: (Optional[str]), Path/URL of the checkpoint from which training is resumed. 
    Could also be one of two special keywords "last" and "hpc". If there is no checkpoint 
    file at the path, an exception is raised. If resuming from mid-epoch checkpoint, 
    training will start from the beginning of the next epoch.
    
    The Lightning Trainer does much more than just “training”. Under the hood, 
    it handles all loop details for you, some examples include:
        *Automatically enabling/disabling grads
        *Running the training, validation and test dataloaders
        *Calling the Callbacks at the appropriate times
        *Putting batches and computations on the correct devices
    '''
    trainer.fit(lpmod, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=valid_dataloader)
    
    
    #%% save best model 
    mname = 'smp_%s_%s.pth' % (net_name, backbone_name)
    mdlpath = os.path.join(root_data_dir, mname.lower())
    torch.save(lpmod.best_model.state_dict(), mdlpath)
    print('Model saved:', mdlpath)
    print('Best score: ', lpmod.best_iou_score)
    print('See model check-points in ' + root_data_dir)
    
    
    
    # %% run validation
    print('\nValidating ...')
    valid_metrics = trainer.validate(lpmod, dataloaders=valid_dataloader, 
                                     verbose=True)
    print(valid_metrics)
    
    # %% plotting training metrics 
    def list_subdirs(rootdir):
        subdirs = []
        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d):
                subdirs.append(d)
        return subdirs 
           
    metric_dir = os.path.join(log_dir, log_name)
    folders = os.listdir(metric_dir)
    #for i in range(len(folders)):
    folders = [os.path.join(metric_dir, d) for d in folders]
    # Sort the folders by modification time
    sorted_folders = sorted(folders, key=lambda f: os.path.getmtime(f))
    # the last one is the folder where the latest 'metrics.csv' resides. 
    #print(sorted_folders)
    print('Plotting ...', sorted_folders[-1])
    plot_csv(sorted_folders[-1], 'metrics.csv')    
    
    print('Finished')    

#%% calling
if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    run(opt)