# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:10:04 2022

@author: renxi
"""
import os, sys
import cv2
import numpy as np

sys.path.append('..')
from pub.colorlabel import label_to_rgbLabel, load_rgb_label, generate_rgb_label
from pub.plot import plot_images
from pub import evam

#----------------------------------------------------------------------------
ROOT_DIR = 'D:\\GeoData\\DLData\\AerialImages'

testimg_dir = os.path.join(ROOT_DIR, 'test', 'images')
testmsk_dir = os.path.join(ROOT_DIR, 'test', 'masks')

pred_dir = os.path.join(ROOT_DIR, 'output', 'test_pred')
predfolders = ['seg4_unet_resnet34_epo200_best.pth',
               'seg4_uplusplus_resnet34_epo200_best.pth',
               'seg4_linknet_resnet34_epo200_best.pth',
               'seg4_fpn_resnet34_epo200_best.pth',
               'seg4_deeplabv3_resnet34_epo200_best.pth',
               'seg4_munet2_epo200_best.pth',
               #'seg_unet_res34_epo5.h5',
               'seg_unet_res34_epo20.h5',
               ]

CLS_DEF_FILE = 'classes.csv'


#-----------------------------------------------------------------------------
plot_dir = os.path.join(ROOT_DIR, 'output', 'test_pred', 'eva_plot')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
 
#load class definition and colors. 
#If the classes.csv file does not exist, a random color table will be generated.
print('Load class information:')
cls_def_file = os.path.join(ROOT_DIR, CLS_DEF_FILE)
if os.path.exists(cls_def_file):
    label_colors = load_rgb_label(cls_def_file)    
else:
    label_colors = generate_rgb_label(n_classes=32)
    print('Not found %s, random color table is used.' % CLS_DEF_FILE)


def disp_eva_mats(mat_list, fi=None):         
    n_mat = len(mat_list)
    #find max colimn and rows of all the matrixes 
    max_rows, max_cols = 0, 0
    for mat in mat_list:
        rows, cols = mat.shape
        max_rows = np.max([rows, max_rows])
        max_cols = np.max([cols, max_cols])
    
    #pad all matrixes to the same shape        
    for i in range(n_mat):
        mat = mat_list[i]
        if mat.shape != (max_rows, max_cols):            
            rr, cc = mat.shape
            mat1 = np.zeros((max_rows,max_cols))
            mat1.fill(np.nan)
            mat1[0:rr,0:cc] = mat
            mat_list[i] = mat1
    
    mt = np.dstack(mat_list)
    
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('-------------------------method----------------------------------', file=fi)
    for i in range(max_rows):
        print('class %d, prec:   ' % i, mt[i][0], file=fi)
        print('class %d, recall: ' % i, mt[i][1], file=fi)
        print('class %d, F1:     ' % i, mt[i][2], file=fi)
        print('---------------------------------------------------------', file=fi)                  
           
    
def disp_ious(iou_list, fi=None):
    nmax = 0
    for iou in iou_list:
        nmax = np.max([nmax, iou.size])
    for i in range(len(iou_list)):
        arr = iou_list[i]        
        if arr.size != nmax:
            size1 = arr.size
            arr1 = np.zeros((nmax,))
            arr1.fill(np.nan)
            arr1[0:size1] = arr
            iou_list[i] = arr1
            
    mt = np.vstack(iou_list)
    print('-------------------------method----------------------------------', file=fi)
    for i in range(mt.shape[1]):
        print('class %d: ' % i, mt[:, i], file=fi)
    print('-----------------------------------------------------------------', file=fi)               



#==========================================================================
logfile = open(os.path.join(plot_dir, 'evalog.txt'), 'w')
    
imgfiles = os.listdir(testimg_dir)
for fn in imgfiles:
    print('\n---'+fn+'-----------------------------')
    bname, ext = os.path.splitext(fn)  
    #load image
    imgpath = os.path.join(testimg_dir, fn)
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #load ground-truth mask 
    gt_mskpath = os.path.join(testmsk_dir, bname+'.png')
    if not os.path.exists(gt_mskpath):
        print('Error: %s does not exist.', gt_mskpath)
        continue    
    gt_msk = cv2.imread(gt_mskpath)[:,:,0]
    
    titles=[]
    disimgs=[]
    disimgs.append(img)
    titles.append('image') 
    disimgs.append(gt_msk)
    titles.append('gt') 
    
    eva_mat_list = []
    ious_list = []
    #load predicted masks 
    #np.set_printoptions(precision=3)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    for mfold in predfolders:
        pth = os.path.join(pred_dir, mfold, 'masks', bname+'.png')
        if not os.path.exists(pth):
            msk = np.random.randint(0,255, img.shape)            
            meaniouv = 0
            print('Error: %s does not exist.' % pth)
        else:
            msk = cv2.imread(pth)[:,:,0]            
            # cal IoU between the gt and the prediction
            iouvs = evam.iou_multi(msk, gt_msk) 
            ious_list.append(iouvs)            
            meaniouv = iouvs.mean()
            maxiouv = iouvs.max()
            miniouv = iouvs.min()
            print('\n'+mfold+':')
            print('Ious:', iouvs)
            print('Ious max min ave : %.3f %.3f %.3f' % (maxiouv, miniouv, meaniouv))
            
            #cal prec, recall, F1
            coeffs = evam.eva_coef_multi(msk, gt_msk)
            #evam.print_eva_coeffs(coeffs)
            eva_mat_list.append(coeffs)
            
        disimgs.append(msk)
        titles.append("Ave IoU:%.3f" % meaniouv)
    
    #display evaluation
    print('\nEvaluation of ' + fn + '...', file=logfile)
    disp_eva_mats(eva_mat_list, fi=logfile)        
    print('IOU', file=logfile)
    disp_ious(ious_list, fi=logfile)
    
    #convert the mask to rgb label image for visualization
    colors = generate_rgb_label(16)
    for i in range(1, len(disimgs)):
        disimgs[i] = label_to_rgbLabel(disimgs[i], label_colors)
    
    plot_images(disimgs, titles, sup_title=fn, 
                save_path=os.path.join(plot_dir, 'plot_'+fn))

    
logfile.close()    
print('Done')


     
    
    
