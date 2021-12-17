import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import dataset
from SynthiaDataset import SynthiaDataset
import augment
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# tensorboard setup
from torch.utils.tensorboard import SummaryWriter

def train(train_epoch, valid_epoch):
    # train model for 40 epochs
    max_score = 0
    min_loss = np.inf
    #EPOCHS = 40
    EPOCHS = 20
    for i in range(0, EPOCHS):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # try writing IoU score to tensorboard
        writer.add_scalar("IoU score train", train_logs["iou_score"], i)
        writer.add_scalar("Loss score train", train_logs["dice_loss"], i)
        writer.add_scalar("IoU score valid", valid_logs["iou_score"], i)
        writer.add_scalar("Loss score valid", valid_logs["dice_loss"], i)
        writer.flush()

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            #torch.save(model, './best_model_fpn.pth')
            torch.save(model, MODEL_NAME)
            print('Model saved!')
        
        # save model with lowest training loss
        if min_loss > train_logs["dice_loss"]:
            min_loss = train_logs["dice_loss"]
            torch.save(model, MODEL_NAME+"_") # add suffix _ to the model
            
        #if i == 25:
        if i == 10:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    writer.close()

# Must for specifying number of worker
if __name__ == "__main__":
    # running scenario
    WRITER_NAME = "runs/deeplab_5"
    writer = SummaryWriter(WRITER_NAME)

    DATA_DIR = "../../Dataset/SYNTHIA-SF/"
    #ENCODER = "se_resnext50_32x4d"
    #ENCODER = "resnet34"
    ENCODER = "efficientnet-b2" # see performance here: https://pytorch.org/vision/stable/models.html
    ENCODER_WEIGHTS = "imagenet"
    #CLASSES = ["car"]
    CLASSES = [
        "void", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", 
        "bus", "train", "motorcycle", "bicycle", "road lines", "other", "road works"
    ] # all synthia dataset class
    #ACTIVATION = "sigmoid" # could be None for logits or 'softmax2d' for multiclass segmentation
    ACTIVATION = "softmax2d"
    DEVICE = "cuda"
    MODEL_NAME = "./all_deeplab_efficientnetb2_iou7.pth"

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

    full_dataset = SynthiaDataset(
        path=DATA_DIR,
        augmentation=augment.get_training_augmentation(), 
        preprocessing=augment.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # TODO: split dataset 70% for training data
    train_size = int(0.7*len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size,valid_size], 
        generator=torch.Generator().manual_seed(1989))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2) # num worker = batch size / 2
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.utils.losses.DiceLoss() # if multiple loss defined, consider using regularization.
    metrics = [
        smp.utils.metrics.IoU(threshold=0.7),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001), # 10^-3 common...decrease by factor of 10 after 10 epoch
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
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    # do training
    train(train_epoch=train_epoch, valid_epoch=valid_epoch)
