import os

from torch.utils.tensorboard.summary import image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import dataset
from SynthiaDataset import SynthiaDataset
import augment
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# tensorboard setup
from torch.utils.tensorboard import SummaryWriter
# running scenario
WRITER_NAME = "runs/deeplab_1"
writer = SummaryWriter(WRITER_NAME)

DATA_DIR = "../../Dataset/SYNTHIA-SF/"
#ENCODER = "se_resnext50_32x4d"
#ENCODER = "resnet34"
ENCODER = "efficientnet-b0" # see performance here: https://pytorch.org/vision/stable/models.html
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["car"]
# CLASSES = [
#     "void", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
#     "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", 
#     "bus", "train", "motorcycle", "bicycle", "road lines", "other", "road works"
# ] # all synthia dataset class
ACTIVATION = "sigmoid" # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = "cuda"

# create segmentation model with pretrained encoder
# model = smp.FPN(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=len(CLASSES), 
#     activation=ACTIVATION,
# )
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = SynthiaDataset(
    augmentation=augment.get_training_augmentation(), 
    preprocessing=augment.get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# inspect model using tensorboard
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
writer.add_graph(model, images)
writer.flush()

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs
max_score = 0
EPOCHS = 40
for i in range(0, EPOCHS):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    
    # try writing IoU score to tensorboard
    writer.add_scalar("IoU score", train_logs["iou_score"], i)
    writer.add_scalar("Loss score", train_logs["dice_loss"], i)
    writer.flush()

    # do something (save model, change lr, etc.)
    if max_score < train_logs['iou_score']:
        max_score = train_logs['iou_score']
        #torch.save(model, './best_model_fpn.pth')
        torch.save(model, './car_deeplab_efficientb3.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

writer.close()