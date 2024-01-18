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
import os
import random
import scipy
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
# Add any python libraries here
def corner_detection(image):
    corner = []
    cmap = []
    for img in image:
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        gray_img= cv2.cornerHarris(gray_img,3,3,0.04)
        img[gray_img>0.02*gray_img.max()] = [0,0,255]
        corner.append(img)
        cmap.append(gray_img)
    return corner,cmap

def ANMS(img,corner,Nbest):
    c_best = []
    lm = peak_local_max(corner,min_distance=15)
    print(lm)#// convert local max values to binary mask
    tot_n = lm.shape[0]
    r = np.zeros(tot_n)
    ed=0
    x=np.zeros((tot_n,1))
    y=np.zeros((tot_n,1))
    for i in range(len(r)):
        r[i] = np.inf
    for i in range(tot_n):
        for j in range(tot_n):
            x_j =lm[j][0]
            y_j =lm[j][1]
            x_i =lm[i][0]
            y_i =lm[i][1]
            if corner[x_j,y_j] > corner[x_i,y_i]:
                ed = (x_j-x_i)**2 + (y_j-y_i)**2
            if ed < r[i]:
                r[i] =ed
                x[i] = x_j
                y[i] = y_j
    xbest = np.zeros(Nbest)
    ybest = np.zeros(Nbest)
    indice = r.argsort()
    indice = np.flip(indice)
    for i in range(Nbest):
        xbest[i] = x[indice[i]]
        ybest[i] = y[indice[i]]
        cv2.circle(img, (int(xbest[i]), int(ybest[i])), 5, (0, 255, 0), -1)
    c_best = np.concatenate((xbest,ybest))
    return img,c_best

def feature_descriptor(img):
    np.zeros(41,)

def sum_sq_dist(vec_1, vec_2):

    sum = 0
    for i in range(len(vec_1)):
        sum += np.square(vec_1[i] - vec_2[i])
    return (sum ** 0.5)

def feature_matching(f_vec1, f_vec2, ratio):
    """
    Each keypoint/feature vector of size 64x1
    Attempts to match the points in two vectors if distance small enough
    Take the ratio of best match to the second best match and if this is below some ratio keep the matched pair or reject it.
    """
    matches = []
    for i in range(len(f_vec1)):
        dist = []
        for j in range(len(f_vec2)):
            dist.append(sum_sq_dist(f_vec1[i], f_vec2[j]))
        sorted(dist)
        idx = np.argsort(dist)
        if dist[idx[0]]/(dist[idx[1]] + 0.001) < ratio:
            matches.append([f_vec1[i], f_vec2[idx[0]]])

    return matches

def RANSAC(matches, n_max):

    for i in range(n_max):
        random_pair = random.choice(matches)
        p1 = random_pair[0]
        p2 = random_pair[1]



    

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    image=[]
    gray_image =[]
    # path = glob.glob("C:/Users/DELL/Downloads/YourDirectoryID_p1/YourDirectoryID_p1/Phase1/Data/Train/Set1/*.jpg")
    # print(path)
    path = os.path.dirname(os.getcwd()) 
    filepath = path + "\\Data\\Train\\Set1\\"
    for i in os.listdir(filepath):
        print(i)
        im = cv2.imread(filepath + i)
        image.append(im)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    corner, cmap = corner_detection(image)
    print(np.shape(corner))
    plt.imsave("img1.png",corner[0])

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    anms = []
    anms_img = []
    Nbest =10
    for i,img in enumerate(corner):
        temp_img,temp = ANMS(img, cmap[i],Nbest)
        anms_img.append(temp_img)
        anms.append(temp)
    print(anms)
    plt.imsave("anms_img.png",anms)

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    matches = feature_matching(kp1, kp2)
    matches = sorted(matches, key = lambda x:x.distance)
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
