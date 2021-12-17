import numpy as np
import matplotlib.pyplot as plt

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

from collections import namedtuple
import numpy as np

class Labels:

    Label_entry = namedtuple("Label_entry", ['name', 'color'],)

    """
    The total number of classes in the training set.
    """
    NUM_CLASSES = 22
    
    """
    ID is equal to index in this array.
        0	-> Void		->   0,0,0
        1	-> Road		->   128,64,128
        2	-> Sidewalk	->   244,35,232
        3	-> Building	->   70,70,70
        4	-> Wall		->   102,102,156
        5	-> Fence	->   190,153,153
        6	-> Pole		->   153,153,153
        7	-> Traffic Light->   250,170,30
        8	-> Traffic Sign	->   220,220,0
        9	-> Vegetation	->   107,142,35
        10	-> Terrain	->   152,251,152
        11	-> Sky		->   70,130,180
        12	-> Person	->   220,20,60
        13	-> Rider	->   255,0,0
        14	-> Car		->   0,0,142
        15	-> Truck	->   0,0,70
        16	-> Bus		->   0,60,100
        17	-> Train	->   0,80,100
        18	-> Motorcycle	->   0,0,230
        19	-> Bicycle	->   119,11,32
        20	-> Road Lines	->   157,234,50
        21	-> Other	->   72,0,98
        22	-> Road Works	->   167,106,29

    """
    COLOURS = [
        Label_entry('void', (0, 0, 0)),                 # ID = 0
        Label_entry('road', (128, 64, 128)),            # ID = 1
        Label_entry('sidewalk', (244, 35, 232)),        # ID = 2
        Label_entry('building', (70, 70, 70)),          # ID = 3
        Label_entry('wall', (102, 102, 156)),           # ID = 4
        Label_entry('fence', (190, 153, 153)),          # ID = 5
        Label_entry('pole', (153, 153, 153)),           # ID = 6
        Label_entry('traffic light', (250, 170, 30)),   # ID = 7
        Label_entry('traffic sign', (220, 220, 0)),     # ID = 8
        Label_entry('vegetation', (107, 142, 35)),      # ID = 9    
        Label_entry('terrain', (152, 251, 152)),        # ID = 10
        Label_entry('sky', (70, 130, 180)),             # ID = 11
        Label_entry('person', (220, 20, 60)),           # ID = 12
        Label_entry('rider', (255, 0, 0)),              # ID = 13
        Label_entry('car', (0, 0, 142)),                # ID = 14
        Label_entry('truck', (0, 0, 70)),               # ID = 15
        Label_entry('bus', (0, 60, 100)),               # ID = 16
        Label_entry('train', (0, 80, 100)),             # ID = 17
        Label_entry('motorcycle', (0, 0, 230)),         # ID = 18
        Label_entry('bicycle', (119, 11, 32)),          # ID = 19
        Label_entry('road lines', (157, 234, 50)),      # ID = 20
        Label_entry('other', (72, 0, 98)),              # ID = 21
        Label_entry('road works', (167, 106, 29)),      # ID = 22
    ]
    """
    DEBUG
    CLASSES = [
        "road", "sidewalk", "building", "pole", "traffic light", 
        "traffic sign", "terrain", "person", "rider", "car", "truck", 
        "bus", "train", "motorcycle", "bicycle"
    ] # all synthia dataset class
    """

    """
    Label is numpy array object!
    """

    @staticmethod
    def colorize(label):
        #Unlabelled pixels are black (zero)!
        colorized = np.zeros((label.shape[0], label.shape[1], 3), dtype = np.uint8)

        for idx in range(len(Labels.COLOURS)):
            colorized[idx == label] = Labels.COLOURS[idx].color

        return colorized  # Numpy array object!

