#import necessary modules
from binhex import REASONABLY_LARGE
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import embedding, optim as optim
# for visualization
from matplotlib import pyplot as plt
import math
import numpy as np

def get_data_loader(img_size=256, batch_size=32):

    # define a transform to 1) scale the images and 2) convert them into tensors
    transform = transforms.Compose([
        transforms.Resize(img_size), # scales the smaller edge of the image to have this size
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder('All-Paintings', transform=transform)

    # load the dataset
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=True
    )

    return train_loader