import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class SynthiaDataset(Dataset):

    CLASSES = [
        "void", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", 
        "bus", "train", "motorcycle", "bicycle", "road lines", "other", "road works"
    ]
    
    def __init__(self, path="../../Dataset/SYNTHIA-SF", classes=None, augmentation=None, preprocessing=None,):
        self.rootdir = Path(path)
        # get list of file
        self.left_imgs, self.left_gts = self.prepare_data(path) 
    
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
        image = cv2.imread(str(self.left_imgs[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.left_gts[index]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
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
        return len(self.left_imgs)

    def prepare_data(self, path):
        """Return list of all images in dataset"""
        left_imgs = list(self.rootdir.glob("*/RGBLeft/*.png"))
        left_gts = list(self.rootdir.glob("*/GTLeft/*.png"))
        
        return left_imgs, left_gts
