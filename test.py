import torch
import augment
import utils
from utils import Labels
import numpy as np
from SynthiaDataset import SynthiaDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

DATA_DIR = "../../Dataset/SYNTHIA-SF/"
#ENCODER = "se_resnext50_32x4d"
ENCODER = "mobilenet_v2"
ENCODER_WEIGHTS = "imagenet"
#CLASSES = ["car"]
CLASSES = [
    "void", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", 
    "bus", "train", "motorcycle", "bicycle", "road lines", "other", "road works"
]
ACTIVATION = "sigmoid" # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = "cuda"

# load best saved checkpoint
#best_model = torch.load('./best_model_fpn.pth')
best_model = torch.load('./car_deeplab_mobilenetv2.pth')
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# create test dataset using full Synthia dataset
test_dataset = SynthiaDataset(
    augmentation=augment.get_validation_augmentation(), 
    preprocessing=augment.get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
# validation metrics
test_dataloader = DataLoader(test_dataset)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.7),
]

# evaluate model on all images in the dataset
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

# test dataset without transformations for image visualization
test_dataset_vis = SynthiaDataset( 
    classes=CLASSES,
)

for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    gt_mask_max = gt_mask.argmax(0)
    gt_mask_color = Labels.colorize(gt_mask_max)
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    #pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = pr_mask.squeeze().cpu().numpy()

    pr_mask_max = pr_mask.argmax(0)
    pr_mask_color = Labels.colorize(pr_mask_max)

        
    utils.visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask_color, 
        predicted_mask=pr_mask_color
    )