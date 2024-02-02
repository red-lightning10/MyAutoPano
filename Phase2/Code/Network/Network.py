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
import torch.nn.functional as F
import kornia
 # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(delta,label,P_B,model):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    criterion = nn.MSELoss()
    if model == "Sup":
        label = label.float()
        delta = delta.float()
        loss = torch.sqrt(criterion(delta, label))
    else:
        label = torch.squeeze(P_B,1)
        loss = torch.sqrt(criterion(delta,label))
    return loss


class HomographyModel(nn.Module):

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss.to(device)}
        return {"loss": loss, "log": logs}

    def validation_step(self,patch_a,patch_b, batch,coordinatesbatch,valcorners,model):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #img_a, patch_a, patch_b, corners, gt = batch
        #delta = self.model(patch_a, patch_b,batch)
        #loss = LossFn(delta, coordinates_batch)
        #return {"val_loss": loss}
        patch = batch
        delta = self(patch_a,patch_b,patch,valcorners)
        loss = LossFn(delta, coordinatesbatch,patch_b,model)
        return {"val_loss": loss.to(device)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}



class SupNet(HomographyModel):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(SupNet, self).__init__()
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
            nn.Conv2d(6, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            # nn.AvgPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            # nn.AvgPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            # nn.AvgPool2d(2,2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
            nn.MaxPool2d(2, 2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            # nn.AvgPool2d(2,2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
        )
        self.layer9 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 8),
            nn.Softmax()
        )
    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################

    def forward(self,xa,xb,x, C_A):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        #############################
        # Fill your network structure of choice here!
        #############################
        return out


class UnsupNet(HomographyModel):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(UnsupNet, self).__init__()
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
            nn.Conv2d(6, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.AvgPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.AvgPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.AvgPool2d(2,2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.AvgPool2d(2,2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AvgPool2d(2,2)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 8),
            nn.Softmax()
        )

    #####

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def TensorDLT(self, H4PT, C_A):
        batch_size = H4PT.size(0)
        H = torch.ones([3, 3], dtype=torch.double)
        H = torch.unsqueeze(H, 0)
        for H4t, ca in zip(H4PT, C_A):

            cb = ca + H4t
            A = []
            B = []

            for i in range(0, 8, 2):
                U = ca[0][i]
                V = ca[0][i + 1]
                U_= cb[0][i]
                V_ = cb[0][i + 1]
                Ai = [[0, 0, 0, -U, V, -1, V_ * U, V_ * V],
                      [U, V, 1, 0, 0, 0, -U_ * U, -U_ * V]]
                # A.append()
                A.extend(Ai)

                bi = [-V_, U_]
                B.extend(bi)
            B = torch.tensor(B)
            B = torch.unsqueeze(B, 1)
            A = torch.tensor(A)
            Ainv = torch.pinverse(A)
            Hi = torch.matmul(Ainv, B)
            H33 = torch.tensor([1])
            Hi = torch.flatten(Hi)
            Hi = torch.cat((Hi, H33), 0)
            Hi = Hi.reshape([3, 3])
            H = torch.cat([H, torch.unsqueeze(Hi, 0)])
        return H[1:65, :, :]
    def dlt(self,C_A, out):
        H= torch.tensor([])
        print(np.shape(C_A))
        print(np.shape(out))
        for i,h4t in enumerate(out):
            C_B = C_A[i] + out[i]
            ca = C_A[i]
            print(type(ca))
            print(type(ca[1][1]))
            A = []
            b = []
            print(type(C_A[1][1]))
            for j in range(4):
                #Ai = np.zeros([2, 8])
                ##Ai[0][3] = -ca[j][2*j]
                #Ai[0][4] = -ca[j][2*j+1]
                #Ai[0][5] = -1
                #Ai[0][6] = C_B[j][2*j+1] * ca[j][2*j]
                #Ai[0[7]] =  C_B[j][2*j+1] * C_B[j][2*j+1]
                #Ai[1][0] = ca[j][2*j]
                #Ai[1][1] = ca[j][2*j+1]
                #Ai[1][2] = 1
                #Ai[1][6] = -C_B[j][2*j] * ca[j][2*j]
                #Ai[1][7] = -C_B[j][2*j+1] * ca[2*j][2*j+1]
                Ai = [[0,0,0,-C_A[i][2*j],-C_A[i][2*j+1],-1,C_B[i][2*j+1]*C_A[i][2*j],C_B[i][2*j+1]*C_A[i][2*j+1]],[C_A[i][2*j],C_A[i][2*j+1],1,0,0,0,-C_B[i][2*j]*C_A[i][2*j],-C_B[i][2*j]*C_A[i][2*j+1]]]
                A.append(Ai)
                bi = np.array[-C_B[i][2*j+1], -C_B[i][2*j]]
                b.append(bi)
                print(A)
            h = torch.dot(torch.inverse(A),b)
            H = torch.cat(H, h.reshape(1, -1), axis=0)
        H = H[1:, :]
        return H

    def stn(self, H, P_A):
        "Spatial transformer network forward function"
        P_A = P_A.unsqueeze(1)
        P_A = P_A.to(torch.double)
        b,c,h,x,y = np.shape(P_A)
        P_A = torch.reshape(P_A,(b,h,x,y))
        out = kornia.geometry.warp_perspective(P_A, H, (64,64), align_corners=True)

        return out

        return x

    def forward(self, xa, xb, x , C_A):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        h = self.layer5(out)
        out = self.TensorDLT(h,C_A)
        out = self.stn(out,xa)
        return out,h
