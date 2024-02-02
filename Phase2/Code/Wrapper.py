#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
# Add any python libraries here
def homography(image,rho):
    C_A = []
    C_B = np.zeros([4,2])# corners of Patch A
    x,y,h = np.shape(image)
    print(x,y)
    p_size = 64  #let patch be 40x40
    center_x,center_y = [np.random.randint(64,x-64),np.random.randint(64,y-64)]

    # selecting center of patchA 110 is 64 + 32sqrt(2)
    C_A.append([center_x-32,center_y+32])
    C_A.append([center_x-32,center_y-32])
    C_A.append([center_x+32,center_y-32])
    C_A.append([center_x+32,center_y+32])
    for i in range(4):
        rad_x= np.random.randint(-rho,rho)
        rad_y = np.random.randint(-rho,rho) #radius for peturbation
        C_B[i][0] = C_A[i][0] +rad_x
        C_B[i][1] = C_A[i][1] +rad_y

    ## NEED TO ADD TRANSLATION ###
    C_A=np.array(C_A,np.float32)
    C_B=np.array(C_B,np.float32)
    P_A = image[center_x-32:center_x+32,center_y-32:center_y+32]
    H_ab = cv2.getPerspectiveTransform(C_A,C_B)
    H_ba = np.linalg.pinv(H_ab)
    I_b = cv2.warpPerspective(image,H_ba,(y,x))
    P_b = I_b[center_x-32:center_x+32,center_y-32:center_y+32]
    H4t = C_B-C_A

    final = np.dstack((P_A,P_b))

    return P_A,P_b,final,H4t,C_A,C_B,I_b

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    rho = 32
    image = []
    path = glob.glob("/home/adhi/YourDirectoryID_p1/Phase2/Train/*.jpg")
    for filename in path:  # assuming gif
        # print("in for")
        im = cv2.imread(filename)
        image.append(im)
    final_stack = []
    h_stack= []
    c_a_stack =[]
    c_b_stack=[]
    count =0
    for i in range(len(image)):
        for j in range(10):
            P_a,P_b,stack,h4t,c_a,c_b,I_b = homography(image[i],rho)
            plt.imsave("/home/adhi/YourDirectoryID_p1/Phase2/Data/Train/A"+str(count)+".jpg",P_a)
            plt.imsave("/home/adhi/YourDirectoryID_p1/Phase2/Data/Train/B" + str(count)+".jpg", P_b)
            h4t = np.reshape(h4t,(1,8))
            c_a = np.reshape(c_a,(1,8))
            c_b = np.reshape(c_b,(1,8))
            h_stack.append(h4t)
            c_a_stack.append(c_a)
            c_b_stack.append(c_b)
            count = count+1
    h_stack=np.array(h_stack)
    c_a_stack =np.array(c_a_stack)
    c_b_stack = np.array(c_b_stack)
    x,y,h = np.shape(h_stack)
    h_stack = np.reshape(h_stack,(x,h))
    c_a_stack = np.reshape(c_a_stack, (x, h))
    c_b_stack = np.reshape(c_b_stack, (x, h))
    print(h_stack)
    DF = pd.DataFrame(h_stack)
    CAF = pd.DataFrame(c_a_stack)
    CBF = pd.DataFrame(c_b_stack)
    DF.to_csv("train_h4t.csv")
    CAF.to_csv("C_A.csv")
    CBF.to_csv("C_B.csv")
    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
