"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
# You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(delta, label):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    criterion = nn.MSELoss()
    label = label.float()
    delta = delta.float()
    loss = torch.sqrt(criterion(delta, label))
    return loss


class HomographyModel(nn.Module):
    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, gt)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch,coordinatesbatch):
        patch = batch
        delta = self(patch)
        loss = LossFn(delta, coordinatesbatch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(HomographyModel):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(Net, self).__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        ...
        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.layer1 = nn.Sequential(
          nn.Conv2d(6,64,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(6),
          nn.ReLU(),
          #nn.AvgPool2d(2,2),
          nn.Conv2d(64,64,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(16),
          nn.ReLU(),
          #nn.AvgPool2d(2,2)
          nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
          nn.Conv2d(64,64,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(6),
          nn.ReLU(),
          #nn.AvgPool2d(2,2),
          nn.Conv2d(64,64,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(16),
          nn.ReLU(),
          #nn.AvgPool2d(2,2)
          nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
          nn.Conv2d(64,128,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(6),
          nn.ReLU(),
          #nn.AvgPool2d(2,2),
          nn.Conv2d(128,128,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(16),
          nn.ReLU(),
          #nn.AvgPool2d(2,2)
          nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
          nn.Conv2d(128,128,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(6),
          nn.ReLU(),
          #nn.AvgPool2d(2,2),
          nn.Conv2d(128,128,kernel_size=(3, 3),stride=1,padding=1),
          #nn.BatchNorm2d(16),
          nn.ReLU(),
          #nn.AvgPool2d(2,2)
        )
        self.layer5 = nn.Sequential(
          nn.Flatten(),
          nn.Linear(8*8*128,1024),
          nn.Sigmoid(),
          nn.Linear(1024,8),
          nn.Softmax()
        )
    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################

    def forward(self,xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #############################
        # Fill your network structure of choice here!
        #############################
        return out
