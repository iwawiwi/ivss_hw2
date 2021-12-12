import matplotlib.pyplot as plt
import numpy as np

import torch as t
import torchvision as tv
import torchvision.transforms as tf

import torch.nn as tnn
import torch.nn.functional as tnf
import torch.optim as top
from torchvision.transforms import transforms

# setup tensorboard
from torch.utils.tensorboard import SummaryWriter

# transforms
transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((.5,), (.5,))
])

# datasets
trainset = tv.datasets.FashionMNIST(
    "./data",
    download=True,
    train=True,
    transform=transform 
)
testset = tv.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transform
)

# dataloaders
trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

# constant for classes
classes = (
    'T-shirt/top', 'Trouser', 'Pullover', 
    'Dress', 'Coat', 'Sandal', 'Shirt', 
    'Sneaker', 'Bag', 'Ankle Boot'
)

# helper function to show image
def plt_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))

class MyNet(tnn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = tnn.Conv2d(1, 6, 5)
        self.conv2 = tnn.Conv2d(6, 16, 5)
        self.pool = tnn.MaxPool2d(2, 2)
        self.fc1 = tnn.Linear(16*4*4, 120)
        self.fc2 = tnn.Linear(120, 84)
        self.fc3 = tnn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(tnf.relu(self.conv1(x)))
        x = self.pool(tnf.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = tnf.relu(self.fc1(x))
        x = tnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = MyNet()

criterion = tnn.CrossEntropyLoss()
optimizer = top.SGD(net.parameters(), lr=.001, momentum=.9)


# default "log_dir" is "runs" and this time we want to be specific
writer = SummaryWriter("runs/fashion_mnist_experiment_1")

# writing to tensorboard
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = tv.utils.make_grid(images)

# show images
plt_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image("four_fashion_mnist_images", img_grid)
writer.flush()