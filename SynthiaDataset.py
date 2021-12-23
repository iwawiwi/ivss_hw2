import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path

class SynthiaDataset(Dataset):

    CLASSES = [
        "void", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", 
        "bus", "train", "motorcycle", "bicycle", "road lines", "other", "road works"
    ]
    
    def __init__(self, path="../../Dataset/SYNTHIA-SF/", classes=None, augmentation=None, preprocessing=None, test=False):
        self.rootdir = Path(path)
        # get list of file
        self.data_imgs, self.data_gts = self.prepare_data(test,path)
        self.test = test

    
        # convert str names to class values on masks
        if classes == None:
            classes = self.CLASSES # if no value, assign all classes
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, index):
        """Get sample from dataset"""
        # read data
        # NOTE: using opencv image reader
        # TODO: should be converted to PIL image?
        image = cv2.imread(str(self.data_imgs[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.data_gts[index]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.test == False:
            image = cv2.resize(image,(864,480),interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask,(864,480),interpolation = cv2.INTER_AREA)
        mask = mask[...,0] # Label on RED channel
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        """Return dataset size"""
        return len(self.data_imgs)

    def prepare_data(self, test, path):
        if test == False:
            data_imgs = list(self.rootdir.glob("*/RGBLeft/*.png"))
            data_gts = list(self.rootdir.glob("*/GTLeft/*.png"))
        
        else :
            data_imgs = list(self.rootdir.glob("*/RGBRight/*.png"))
            data_gts = list(self.rootdir.glob("*/GTright/*.png"))

        return data_imgs, data_gts
