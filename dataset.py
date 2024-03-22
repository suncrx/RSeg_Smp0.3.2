
# import the necessary packages
import os, sys
import random
import cv2
import numpy as np
#import copy

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataug import get_train_aug, get_val_aug

'''
torch.utils.data.Dataset is an abstract class representing a dataset. 
Your custom dataset should inherit Dataset and override the following 
methods:

    __len__ : so that len(dataset) returns the size of the dataset.

    __getitem__: to support the indexing such that dataset[i] can be 
                 used to get i-th sample.

'''

# data transformation
#IMG_TRANS = transforms.Compose([transforms.ToTensor(),
                                #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
#                                ])   



def make_one_hot(inp, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    inp = inp.long()
    shape = np.array(inp.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, inp.cpu(), 1)

    return result


# Dataset for segmentation
# return numpy image (H, W, 3)
# and numpy mask (H, W)
class SegDataset(Dataset):
    def __init__(self, root_dir, mode="train", 
                 n_classes=1, imgH=256, imgW=256,
                 apply_aug = False, sub_size=-1):
        
        assert mode in {"train", "val", "test"}
        self.mode = mode
        
        self.root = root_dir
        
        self.imgW = imgW
        self.imgH = imgH        
        
        # binary segmentation : n_classes = 1
        # multi-class segmentation : n_classes > 1
        self.n_classes = n_classes
        
        
        if apply_aug:
            if mode == 'train':
                self.aug = get_train_aug(height=imgH, width=imgW)
            else:
                self.aug = get_val_aug(height=imgH, width=imgW)
        else:
            self.aug = None
        
            
        #self.trans = IMG_TRANS
        
        # search image and mask filepaths
        self.images_directory = os.path.join(self.root, mode, "images")
        self.masks_directory = os.path.join(self.root, mode, "labels")
        #assert os.path.exists(self.images_directory)
        #assert os.path.exists(self.images_directory)
        if not os.path.exists(self.images_directory):
            print("ERROR: Cannot find directory " + self.images_directory)
            sys.exit()
            
        #if not os.path.exists(self.masks_directory):
        #    print("ERROR: Cannot find directory " + self.masks_directory)
        #    sys.exit()            

        print('Scanning files in %s ... ' % self.mode)
        print(' ' + self.images_directory)
        print(' ' + self.masks_directory)        
        self.imgPairs = self._list_files()
        
        
        #subset the dataset
        #randomly select num items
        if sub_size > 0 and sub_size <= 1:
            num = np.int32(len(self.imgPairs)*sub_size)
            self.imgPairs = random.sample(self.imgPairs, num)
        elif sub_size > 1:
            num = min(len(self.imgPairs), np.int32(sub_size))
            self.imgPairs = random.sample(self.imgPairs, num)
        
        print(" #image pairs: ", len(self.imgPairs))



    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imgPairs)


    # return a dictionry 
    # {'image':image, 'mask':mask, 'oimage':oimage, 'omask':omask}  
    # image: tensor image with shape (3, H, W), and data range (0 ~ 1.0)
    # mask: binary mask image of size (H, W), with value 0 and 1.0.
    # oimage: uint8 image numpy (H, W, 3)
    # omask:  uint8 mask numpy (H, W)
    def __getitem__(self, idx):
        
        oimage, omask = self.get_image_and_mask(idx)

        # resize image and mask if necessary        
        if (self.imgW is not None) and (self.imgH is not None):            
            if oimage.shape[0:2] != (self.imgH, self.imgW):
                oimage = cv2.resize(oimage, (self.imgW, self.imgH),
                                   interpolation=cv2.INTER_NEAREST)            
            if omask.shape != (self.imgH, self.imgW):
                omask = cv2.resize(omask, (self.imgW, self.imgH), 
                                  interpolation=cv2.INTER_NEAREST)           
        
        # apply augmentation
        #image, mask = copy.deepcopy(oimage), copy.deepcopy(omask)
        image, mask = oimage, omask
        if self.aug is not None:
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
       
        # convert image to tensor
        #image = self.trans(image)
        image = transforms.ToTensor()(image)
        
        # transform mask to tensor
        # binary segmentation            
        if self.n_classes <= 1:
            # convert to (0, 1) float 
            mask = transforms.ToTensor()(mask > 0)
            mask = mask.float()
            #return {'image':image, 'mask':mask, 
            #        'oimage':oimage, 'omask':omask}        
            return (image, mask)
        # multi-class segmentation
        else:           
            # convert label mask to one-hot tensor
            #masks = [(mask == v) for v in range(self.n_classes)]
            #mask_cat = np.stack(masks, axis=0).astype('long')
            mask = np.expand_dims(mask, axis=0)
            mask = np.expand_dims(mask, axis=0)
            mask_cat = make_one_hot(torch.as_tensor(mask), self.n_classes)
            mask_cat = torch.squeeze(mask_cat)
            #return {'image':image, 'mask':mask_cat, 
            #        'oimage':oimage, 'omask':omask}            
            return (image, mask_cat)
    
    def get_image_and_mask_path(self, idx):
        return (self.imgPairs[idx]['image'], self.imgPairs[idx]['mask'])
    

    def get_image_and_mask(self, idx):
        # grab the image and mask path from the current index
        imagePath = self.imgPairs[idx]['image']
        maskPath = self.imgPairs[idx]['mask']
        
        # load the image from disk, swap its channels from BGR to RGB,
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
        # read the associated mask from disk in grayscale mode
        if os.path.exists(maskPath):
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        
        return image, mask
    
    
    def get_image_filepaths(self):
        return [item['image'] for item in self.imgPairs]
    
    
    def _list_files(self):
        #EXTS = ['.png', '.bmp', '.gif', '.jpg', '.jpeg']
        imgs = os.listdir(self.images_directory)
        if os.path.exists(self.masks_directory):
            msks = os.listdir(self.masks_directory)
        else:
            msks = []
        
        #extract mask file names and extensions
        msk_names = []
        msk_exts = []
        for i in range(len(msks)):
            path = os.path.join(self.masks_directory, msks[i])
            if not os.path.isfile(path):
                continue
            fname, ext = os.path.splitext(msks[i])
            msk_names.append(fname)
            msk_exts.append(ext)
            
        # extract image and mask pairs        
        imgpaths = []
        mskpaths = []
        for i in range(len(imgs)):
            path_img = os.path.join(self.images_directory, imgs[i])
            if not os.path.isfile(path_img):
                continue
            
            fname, ext = os.path.splitext(imgs[i])
            # if finding a matched mask file in msk_names
            if fname in msk_names:
                idx = msk_names.index(fname)
                path_msk = os.path.join(self.masks_directory, fname + msk_exts[idx])
            # or not, generate a virtual filepath for the mask file
            else:
                path_msk = os.path.join(self.masks_directory, fname + '.png')
                print('Warning: Cannot find mask file %s' % path_msk)
            
            imgpaths.append(path_img)
            mskpaths.append(path_msk)                        
        
        #make image pairs list
        imgPairs = [{'image':fp1, 'mask':fp2} for fp1, fp2 in zip(imgpaths, mskpaths)]    
        
        return imgPairs                                  


# check data integrity
def check_data(root_dir):
    subdirs = ['train', 'val']
    for sd in subdirs:
        img_dir = os.path.join(root_dir, sd, 'images') 
        msk_dir = os.path.join(root_dir, sd, 'masks') 
    
        if not os.path.exists(img_dir):
            print('ERROR: '+img_dir+' does not exist.')
            return False
        if not os.path.exists(msk_dir):
            print('ERROR: '+msk_dir+' does not exist.')
            return False

    return True    

    
if __name__ == '__main__':
    import matplotlib.pylab as plt

    data_dir = 'D:\\GeoData\\DLData\\buildings'      
    ds = SegDataset(data_dir, 'train', 1, 256, 256,apply_aug=True,sub_size=32)        
    for i in range(10):        
        samp = ds[i]
        #oimg, omsk = samp['oimage'], samp['omask']
        img, msk = samp
        
        #print('original image: ', oimg.shape, omsk.shape)
        print('transformed image: ', img.shape, msk.shape)
        
        plt.figure()
        #plt.subplot(221)        
        #plt.imshow(img)        
        #plt.subplot(222)        
        #plt.imshow(msk)                
        plt.subplot(121)        
        plt.imshow(np.moveaxis(img.numpy(), 0, -1))        
        plt.subplot(122)        
        plt.imshow(torch.squeeze(msk).numpy())             
        plt.show()

'''        
    ds = SegDataset(data_dir, 'train', 6, 256, 256, apply_aug=True)
    for i in range(10):
        img, msk_cat = ds[i]['image'], ds[i]['mask']
        print('image and one-hot labels: ', img.shape, msk_cat.shape)
        
        plt.subplot(121)        
        plt.imshow(np.moveaxis(img.numpy(), 0, -1))
        
        plt.subplot(122)        
        plt.imshow(np.dstack([msk_cat[0],msk_cat[1], msk_cat[2]]))
        
        plt.show()        
        '''
        