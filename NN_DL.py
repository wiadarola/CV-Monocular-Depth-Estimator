import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import glob

class DepthDataset(Dataset):
    def __init__(self, rgb, depth):
        self.input = rgb
        self.output = depth

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_path = self.input[index]
        output_path = self.output[index]

        with Image.open(input_path) as input_image, Image.open(output_path) as output_image:
            rgb = torch.tensor(np.array(input_image) / 255, dtype=torch.float).reshape(3,480,640)
            depth = torch.tensor(np.array(output_image), dtype=torch.float)

        return rgb, depth
    
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Initialize the weights of conv layers
        init.xavier_uniform_(self.conv1.weight.data)
        init.xavier_uniform_(self.conv2.weight.data)
        init.xavier_uniform_(self.conv3.weight.data)
        init.xavier_uniform_(self.conv4.weight.data)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.upsample(out)
        out = self.conv4(out)
        return out
    
def data_load(params):
    rgb = glob.glob('./sync/**/rgb*', recursive = True)
    depth = glob.glob('./sync/**/sync*', recursive = True)
    rgb.sort()
    depth.sort()

    X_train, X_test, y_train, y_test  = train_test_split(rgb, depth, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


    train_loader = DataLoader(DepthDataset(X_train, y_train), **params)
    val_loader = DataLoader(DepthDataset(X_val, y_val), **params)
    test_loader = DataLoader(DepthDataset(X_test, y_test), **params)

    return train_loader, val_loader, test_loader
