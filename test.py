# USAGE
# python test.py
#

# import the necessary packages
# %% import installed packages
import os
import sys
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib.pylab as plt

import segmentation_models_pytorch as smp
# explictly import utils if segmentation_models_pytorch >= 0.3.2

import models
from dataset import SegDataset
from utils.plot import plot_prediction


#%% get current directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # FasterRCNN root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# determine the device to be used for training and evaluation
DEV = "cuda" if torch.cuda.is_available() else "cpu"

#======================================================================
def make_prediction(model, image, out_H, out_W, binary=False, conf=0.5):  
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad(): 
        # apply image transformation. This step turns the image into a tensor.
        # with the shape (1, 3, H, W). See IMG_TRANS in dataset.py
        #image = IMG_TRANS(image)
        image = torch.unsqueeze(image, 0)        
        
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred = model.forward(image).squeeze()                
        
        # Sigmod or softmax has been performed in the net
        if binary:
            pred = cv2.resize(pred.numpy(), (out_W, out_H))
            pred = np.uint8(pred>=conf)
        else:
            #determine the class by the index with the maximum                     
            pred = np.uint8(torch.argmax(pred, dim=0))        
            #resize to the original size        
            pred = cv2.resize(pred, (out_W, out_H),
                                   interpolation=cv2.INTER_NEAREST)
            print('Found classes: ', np.unique(pred))                  
        
    return pred


def Cal_IoU(predMask, gtMask, n_classes):
    if n_classes>1:
        pm = [predMask==v for v in range(n_classes)]
        gm = [gtMask==v for v in range(n_classes)]
        pm = np.array(pm)
        gm = np.array(gm)
    else:
        pm = np.uint8(predMask>0)
        gm = np.uint8(gtMask>0)
        
    iou = smp.utils.metrics.IoU()
    iouv = iou.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    return iouv


def run(opt):
    #print(opt)
    # get parameters
    data_dir, img_sz = opt.data_dir, opt.img_sz
    model_file = opt.model_file
    out_dir = opt.out_dir
    conf = opt.conf
    
    if not os.path.exists(model_file):
        raise Exception('Can not find model path: %s' % model_file)
    
    # make output folders
    model_basename = os.path.basename(model_file)
    # make output folder for predicting images
    os.makedirs(os.path.join(data_dir, 'out'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'out', 'test_pred'), exist_ok=True)
    outpred_dir = os.path.join(data_dir, 'out', 'test_pred', model_basename)
    if os.path.exists(outpred_dir):
        shutil.rmtree(outpred_dir)
    os.makedirs(outpred_dir)
        
    #--------------------------------------------------------------------    
    # load our model from disk and flash it to the current device
    print("Loading model: %s" % model_file)
    model, model_name, n_classes, class_names = models.utils.load_seg_model(model_file)    
    
    #---------------------------------------------------------------------------
    # load the image paths in our testing directory and
    # randomly select 10 image paths
    print("Loading test image ...")
    testDS = SegDataset(data_dir, mode="test", 
                        n_classes=n_classes, 
                        imgH=img_sz, imgW=img_sz,
                        apply_aug = False)
    
    nm = min(len(testDS), 10)
    IoUs = []
    for i in range(nm):
        #get preprocessed image and mask
        img, gtMask =  testDS[i]    
        #get the original image and mask
        ori_img, ori_gtMask = testDS.get_image_and_mask(i)
        #get the image and mask filepaths
        imgPath, mskPath = testDS.get_image_and_mask_path(i)
        
        outH, outW = ori_img.shape[0:2]
        
        # make predictions and visualize the results
        print('\nPredicting ' + imgPath)    
        #for binary segmentation, pred is a uint8-type mask with 0, 1;
        #for multi-class segmentation, pred is a uint8-type mask with
        #class labels: 0, 1, 2, 3, ... , n_class-1
        is_binary = (n_classes<2)
        pred = make_prediction(model, img, outH, outW, 
                               binary=is_binary, conf=conf)
        
        #IoU evaluation
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        
        #if config.N_CLASSES>1:
        #    iouvs = evam.iou_multi(pred, ori_gtMask)
        #    print('IoUs:', iouvs)
        #    print('IoUs max min ave : %.3f %.3f %.3f' % (iouvs.max(), iouvs.min(), iouvs.mean()))
        #else:
        #    iouv = evam.iou_binary(pred, ori_gtMask)
        #    print('IoUv:', iouv)
    
        iouv = Cal_IoU(pred, ori_gtMask, n_classes=n_classes)    
        print('IoU: %.3f' % iouv)
        IoUs.append(iouv)
        
        #------------------------------------
        #save and convert results to rgb label for visualization 
        image_basename = os.path.basename(imgPath) 
        bname, ext = os.path.splitext(image_basename)
        out_mskPath = os.path.join(outpred_dir, bname+'.png')        
        fig_path = os.path.join(outpred_dir, 'plot_'+bname+'.png')           
        stitle = '%s IoU %.3f' % (image_basename, iouv)
        if is_binary:
            #save predicted mask (one channel)    
            Mask = np.uint8(pred*255)        
            cv2.imwrite(out_mskPath, Mask)
            #plot results
            plot_prediction(ori_img, ori_gtMask, Mask, 
                            sup_title=stitle, save_path=fig_path, 
                            auto_close=True)   
        else:
            Mask = np.uint8(pred)        
            cv2.imwrite(out_mskPath, Mask)
            Mask_rgb = np.uint8(pred*255)        
            out_rgbMskPath = os.path.join(outpred_dir, bname+'_rgb.png')        
            cv2.imwrite(out_rgbMskPath, Mask_rgb)
            ori_gtMask_rgb = np.uint8(ori_gtMask*255)
            #plt results
            plot_prediction(ori_img, ori_gtMask_rgb, Mask_rgb, 
                            sup_title=stitle, save_path=fig_path, 
                            auto_close=True)   
        
        #copy original image and mask to the output folder
        mask_basename = os.path.basename(mskPath)
        bname, ext = os.path.splitext(mask_basename)         
        shutil.copy(mskPath, os.path.join(outpred_dir, bname+'_gt'+ext))        
        shutil.copy(imgPath, os.path.join(outpred_dir, image_basename))
        
        '''
        #convert Mask to rgb label image and save
        RgbMask = label_to_rgbLabel(Mask, label_colors)
        BgrMask = cv2.cvtColor(RgbMask, cv2.COLOR_RGB2BGR)            
        cv2.imwrite(out_rgbMskPath, BgrMask)
        print('Saved: %s' % out_rgbMskPath)
        
        #convert ground-truth mask to rgb label image
        gtRgbMask = label_to_rgbLabel(ori_gtMask, label_colors)            
        '''
        
    #IouS 
    MIoUs = np.array(IoUs)
    print('\nMax IoU: %.3f' % MIoUs.max())
    print('Min IoU: %.3f' % MIoUs.min())
    print('Mean IoU: %.3f' % MIoUs.mean())
    plt.figure()
    plt.plot(MIoUs, '.')
    plt.show()
        
    print('Done!')    
    print('Results saved: %s' % outpred_dir)       
    
    

#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str, 
                        default='D:/GeoData/DLData/Buildings/out/unet_resnet34_best.pt', 
                        help='model filepath')
    
    parser.add_argument('--data_dir', type=str, 
                        default='D:/GeoData/DLData/Buildings', 
                        help='test image directory')
    
    parser.add_argument('--img_sz', type=int, 
                        default=512, help='input image size (pixels)')
    
    parser.add_argument('--out_dir', type=str, default=ROOT / 'out', 
                        help='training output path')    
    
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='test confidence')    
               
    return parser.parse_args() 


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
    