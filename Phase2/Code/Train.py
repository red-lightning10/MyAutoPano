#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel,LossFn,SupNet,UnsupNet
import sys
import os
import numpy as np
import random
import skimage
import PIL
from PIL import Image
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import pandas as pd

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize,C_A):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []
    I2Batch = []
    Stack =[]
    ImageNum = 0
    corners = []
    while ImageNum < MiniBatchSize:
        # Generate random image
        lst = os.listdir(BasePath)
        RandIdx = random.randint(0, (len(lst)//2-1))
        RandImageName1 = BasePath + os.sep + "A"+str(RandIdx) + ".jpg"
        RandImageName2 = BasePath + os.sep + "B"+str(RandIdx) + ".jpg"
        ImageNum += 1
        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        print(RandImageName1)
        I1 = np.float32(Image.open(RandImageName1))/255
        print(I1)
        I2 = np.float32(Image.open(RandImageName2))/255
        final = np.dstack((I1, I2))
        print(np.shape(final))
        final = np.reshape(final,(6,64,64))
        I1 = np.reshape(I1,(3,64,64))
        I2 = np.reshape(I2,(3,64,64))
        Coordinates = TrainCoordinates[RandIdx]
        # Append All Images and Mask
        corners.append(torch.from_numpy(C_A[RandIdx, :]))
        I1Batch.append(torch.from_numpy(I1))
        I2Batch.append(torch.from_numpy(I2))
        Stack.append(torch.from_numpy(final))
        CoordinatesBatch.append(torch.tensor(Coordinates))

    return torch.stack(I1Batch), torch.stack(I2Batch),torch.stack(Stack),torch.stack(CoordinatesBatch),torch.stack(corners)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
    ValCoordinates,
    C_A
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    if ModelType=="Sup":
        model = SupNet(6*64*64,8)
    else:
        model = UnsupNet(6*64*64,8)
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    valloss= []
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, I2Batch, StackBatch,CoordinatesBatch,corners = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize,C_A
            )
            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch,I2Batch,StackBatch,corners)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch,I2Batch,ModelType)
            print(LossThisBatch)
            Optimizer.zero_grad()
            LossThisBatch = x = torch.tensor([1.], requires_grad=True)
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")
            model.eval()
            P_A,P_B, valBatch,valcoorBatch,valcorners = GenerateBatch(
               "/home/tshanmugam/CV2/Phase2/Data/Val", DirNamesTrain, ValCoordinates, ImageSize, MiniBatchSize,C_A
            )
            result = model.validation_step(P_A,P_B,valBatch,valcoorBatch,valcorners,ModelType)
            valloss.append(result)
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
        DF = pd.DataFrame(valloss)
        DF.to_csv("valloss.csv")
def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/home/tshanmugam/CV2/Phase2/Data/Train",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="/home/tshanmugam/CV2/Phase2/Code3/Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=19,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)
    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)
    ValCoordinates = ReadLabels("val_h4t.csv")
    C_A = ReadLabels("C_A.csv")
    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
        ValCoordinates,
        C_A
    )


if __name__ == "__main__":
    main()
