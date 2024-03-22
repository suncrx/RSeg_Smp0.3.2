# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:06:47 2023

@author: renxi
"""
# import the necessary packages
import torch
import segmentation_models_pytorch as smp

from .munet1 import MUnet1
from .munet2 import MUnet2

#import munet1
#import munet2

#-----------------------------------------------------------------------------
def create_model(arct='unet', encoder='resnet34', n_classes=1, in_channels=3):

    m_fullname = arct + '_' + encoder     
    
    activation_name = 'softmax' if n_classes>1 else "sigmoid"
    
    ENCODER_WEIGHTS = 'imagenet'
    ENCODER = encoder
    
    if arct.lower()=='unet':
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 
        # use `imagenet` pre-trained weights for encoder initialization
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # model output channels (number of classes in your dataset)
        MODEL = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)
                         #activation='softmax')
                                 
    elif arct.lower()=='unetplusplus':
        MODEL = smp.UnetPlusPlus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)

                         
    elif arct.lower()=='linknet':
        MODEL = smp.Linknet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)
                         #activation='softmax')
    
    elif arct.lower()=='fpn':
        MODEL = smp.FPN('resnet34', encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)
    
    elif arct.lower()=='deeplabv3':
        MODEL = smp.DeepLabV3('resnet34', encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)
        
    elif arct.lower()=='munet1':        
        MODEL = MUnet1(n_classes=n_classes, activation=activation_name)
        m_fullname = arct
        
    elif arct.lower()=='munet2':        
        MODEL = MUnet2(n_classes=n_classes, activation=activation_name)
        m_fullname = arct

    else:
        MODEL = None
        m_fullname = 'None'
        
    return MODEL, m_fullname        


# generate a model file name 
def generate_model_filename(model_name):
    mfname = 'seg4_%s_best.pth' % model_name  
    return mfname


# save model and auxillary information
def save_seg_model(model, fpath, arct, encoder, 
                   n_classes, names, in_channels):
    torch.save({
            'n_classes': n_classes,
            'class_names': names, 
            'in_channels': in_channels,
            'arct': arct,
            'encoder': encoder,
            'model_state_dict': model.state_dict(),                        
            },  fpath)


def load_seg_model(fpath):
    # load the model and the trained weights
    mdict = torch.load(fpath)
   
    n_classes = mdict['n_classes']
    class_names = mdict['class_names']
    in_channels = mdict['in_channels']
   
    arct = mdict['arct']
    encoder = mdict['encoder']
    
    model, model_name = create_model(arct=arct, encoder=encoder, 
                         n_classes=n_classes, 
                         in_channels=in_channels)
    model.load_state_dict(mdict['model_state_dict']) #, map_location=DEVICE)
    
    return model, model_name, n_classes, class_names



if __name__ == '__main__':
    m, mname = create_model(model_name='unetplusplus')
    print(m)
    print(mname)
    mfname = generate_model_filename(mname)
    print(mfname)