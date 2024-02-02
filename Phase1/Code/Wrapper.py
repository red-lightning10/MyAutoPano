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

import numpy as np
import cv2
import glob
import scipy
import os
import random
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import itertools

# Add any python libraries here


def corner_detection(image, method = "Harris"):

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    img = image.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "Harris":
        corners = cv2.cornerHarris(gray_img, blockSize=3, ksize=3, k=0.01)

        corners[corners < 0.001 * corners.max()] = 0
        corner_indices = np.argwhere(corners > 0.1 * corners.max())
        for x,y in corner_indices:
            cv2.circle(img, (y, x), 1, (0, 0, 255), -1)
    
    if method == "Shi-Tomasi":
        corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=300, qualityLevel=0.05, minDistance=10)
        corners = np.int0(corners).reshape(-1, 2)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x,y), 2, (0, 0, 255), -1)

    return img, corners

def ANMS(image, corner, Nbest):
    
        
    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    img = image.copy()
    c_best = []
    corner = np.array(corner)
    lm = peak_local_max(corner, min_distance = 5) # convert local max values to binary mask
    tot_n = lm.shape[0]
    # print(tot_n)
    if Nbest > tot_n:
        Nbest = tot_n

    r = np.zeros(tot_n)
    ed = 0
    x = np.zeros((tot_n, 1))
    y = np.zeros((tot_n, 1))
    for i in range(tot_n):
        r[i] = np.inf
        for j in range(tot_n):
            x_j =lm[j][1]
            y_j =lm[j][0]
            x_i =lm[i][1]
            y_i =lm[i][0]
            
            if corner[y_i, x_i] < corner[y_j, x_j]:
                ed = np.square(x_j - x_i) + np.square(y_j - y_i)
            
            if ed < r[i]:
                r[i] = ed
                x[i] = x_j
                y[i] = y_j
    
    xbest = np.zeros(Nbest)
    ybest = np.zeros(Nbest)
    indice = r.argsort()
    indice = np.flip(indice)

    for i in range(Nbest):
        idx = indice[i]
        xbest[i] = x[idx]
        ybest[i] = y[idx]
        
        cv2.circle(img, (int(xbest[i]), int(ybest[i])), 2, (0, 255, 0), -1)
        c_best.append((xbest[i], ybest[i]))
    
    return img, c_best
    
def feature_descriptor(image, c_best):

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
    
    descriptor = []
    img = image.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p_size = 41
    gray_img = cv2.copyMakeBorder(gray_img, 20,20,20,20, cv2.BORDER_CONSTANT)

    for i, j in c_best :
        
        i = int(i) + 20
        j = int(j) + 20

        f_img = gray_img[j - p_size//2 : j + p_size//2, i - p_size//2 : i + p_size//2]
        blur_img = cv2.GaussianBlur(f_img, (5,5), 0)
        f_vec = cv2.resize(blur_img, (8,8), interpolation = cv2.INTER_AREA)
        f_vec = (f_vec - f_vec.mean()) / np.std(f_vec)

        f_vec = f_vec.flatten()
        descriptor.append(f_vec)

    return descriptor

def sum_sq_dist(vec_1, vec_2):

    sum = 0

    for i in range(len(vec_1)):
        sum += np.square(vec_1[i] - vec_2[i])
    
    return np.sum(sum)

def feature_matching(f_vec1, f_vec2, corners_1, corners_2, ratio):
    """
	Feature Matching
	Save Feature Matching output as matching.png

    Each keypoint/feature vector of size 64x1
    Attempts to match the points in two vectors if distance small enough
    Take the ratio of best match to the second best match and if this is below some ratio keep the matched pair or reject it.
    """
    good_matches = []
    matches = []
    match_dist = []
    for i in range(len(f_vec1)):
        dist = []
        for j in range(len(f_vec2)):
            dist.append(sum_sq_dist(f_vec1[i], f_vec2[j]))
        idx = np.argsort(dist)
        
        if dist[idx[0]]/(dist[idx[1]] + 0.0001) < ratio:

            keypoints_1 = ((int(corners_1[i][0]), int(corners_1[i][1])))
            keypoints_2 = ((int(corners_2[idx[0]][0]), int(corners_2[idx[0]][1])))
            matches.append((keypoints_1, keypoints_2))
            match_dist.append(dist[idx[0]])
    
    dist_idx = np.argsort(match_dist)
    # print(dist_idx)
    # print(matches)
    for i in range(len(dist_idx)):
        good_matches.append(matches[dist_idx[i]])

    return keypoints_1, keypoints_2, good_matches

def plot_feature_correspondance(source, target, matches):

    concatenated_image = np.concatenate((source, target), axis = 1)
    
    corners_s = matches[:, 0]
    corners_t  = matches[:, 1]

    for (x_1, y_1) , (x_2, y_2) in zip(corners_s, corners_t):
        cv2.circle(concatenated_image, (int(x_1), int(y_1)), 2, (0, 255, 0), -1)
        cv2.circle(concatenated_image, (int(x_2 + source.shape[1]), int(y_2)), 2, (0, 255, 0), -1)
        cv2.line(concatenated_image, (x_1, y_1), (x_2 + source.shape[1], y_2 ), (200, 0, 0), 1)
    
    return concatenated_image
    
def RANSAC(matches, n_max, threshold):

    
    """
	Refine: RANSAC, Estimate Homography
	"""
    
    inliers = []
    h_matrix = []
    num_max_inliers = 0

    for i in range(n_max):
        
        choices = np.random.choice(np.arange(np.shape(matches)[0]), 4, replace=False)
        
        p1 = matches[choices, 0]
        p1_prime = matches[choices, 1]

        h_cap = cv2.getPerspectiveTransform(np.float32(p1), np.float32(p1_prime))

        p1_transformed = [np.matmul(h_cap, [x, y, 1])  for x,y in matches[:, 0, :]]
        p1_transformed = np.array(p1_transformed)

        p1_transformed_error_calc = np.zeros_like(p1_transformed)
        p1_transformed_error_calc[:,0] = p1_transformed[:,0]/(p1_transformed[:,2] + 0.001)
        p1_transformed_error_calc[:,1] = p1_transformed[:,1]/(p1_transformed[:,2] + 0.001)
        

        inliers_check = [sum_sq_dist(matches[j, 1, :], p1_transformed_error_calc[j]) < threshold for j in range(np.shape(matches)[0])]
        num_inliers = np.sum(inliers_check)

        if num_inliers > num_max_inliers:

            inliers_indices = np.where(inliers_check)
            h_matrix = h_cap
            num_max_inliers = num_inliers
            ratio_inliers = num_max_inliers / np.shape(matches)[0]

    inliers = matches[inliers_indices]
    # print(num_max_inliers)
       
    ## recompute with inliers
    # if len(inliers) > 3:
    #     choices = np.random.choice(np.arange(np.shape(inliers)[0]), 4, replace=False)
    #     h_matrix = cv2.getPerspectiveTransform(np.float32(inliers[choices, 0, :]), np.float32(inliers[choices, 1, :]))
        
    return h_matrix, inliers, ratio_inliers


def stitch_the_image(source, target, h_matrix):

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    corners_source = np.float32([[0, 0], [0, np.shape(source)[0]], [np.shape(source)[1], np.shape(source)[0]], [np.shape(source)[1], 0]]).reshape(-1, 1, 2)
    corners_target = np.float32([[0, 0], [0, np.shape(target)[0]], [np.shape(target)[1], np.shape(target)[0]], [np.shape(target)[1], 0]]).reshape(-1, 1, 2)
    
    warped_corners = cv2.perspectiveTransform(corners_source, h_matrix)

    corners = np.concatenate((warped_corners, corners_target), axis=0)

    [x_min, y_min] = np.int32(corners.min(axis=0).flatten())
    [x_max, y_max] = np.int32(corners.max(axis=0).flatten())

    h_translation = np.array([[1, 0, - x_min], [0, 1, - y_min], [0, 0, 1]])

    H = np.matmul(h_translation, h_matrix)

    warped_image = cv2.warpPerspective(source, H, (x_max - x_min + 1, y_max - y_min + 1))
    warped_image[- y_min : np.shape(target)[0] - y_min, - x_min : np.shape(target)[1] - x_min] = target
    # print(- x_min, np.shape(target)[1] - x_min, - y_min, np.shape(target)[0] - y_min)

    return warped_image

# def blend_the_image(source, result_image, method = "simple"):

#     if method == "simple":
#         mask = np.where(result_image, 1, 0)
#         alpha = np.float32(mask)
        
#         source = cv2.resize(source, result_image.shape[1::-1])
#         blend_image = result_image * alpha + cv2.resize(source, result_image.shape[1::-1]) * (1 - alpha)
#         blend_image = np.uint8(blend_image)
        
#     return blend_image

#     # dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

def many_to_one_matches_check(inliers):

    # for i,j in inliers:
    # corner1_unique, count1 = np.unique(inliers[:, 0, :], return_counts=True)
    # corner2_unique, count2 = np.unique(inliers[:, :, :], return_counts=True)

    # print(corner2_unique, count2)
    inlier_unique = []
    inlier_unique = [i for i in dict(inliers[:, 1, :]).items() if i not in inlier_unique]
    # print(len(inlier_unique), len(inliers))

    if len(inlier_unique) < 0.85 * len(inliers):
        print("Abort stitching, many-to-one matches found")
        return True

    else:
        return False

def return_no_matches():
    return np.zeros((3,3)), 0, 0

def corner_pipeline(image, i, results_path, corner_method = "Harris"):

    corner, cmap = corner_detection(image, corner_method)
    plt.imsave(results_path + "new_image_after_corner_detection_" + str(i+1) + ".png", corner)

    Nbest = 300

    if corner_method == "Harris":
        temp_img, temp = ANMS(image, cmap, Nbest)
        corner_unique = []
        corner_unique = [j for j in dict(temp).items() if j not in corner_unique]
        # corners.append(corner_unique)
        plt.imsave(results_path + "new_image_after_ANMS_" + str(i+1) + ".png" , temp_img)

    else:
        corners.append(cmap)

    corners = corner_unique
    descriptor = feature_descriptor(image, corners)
    # descriptors.append(descriptor)
        
    return corners, descriptor

def matching_pipeline(image1, image2, i, j, corners1, corners2, descriptor1, descriptor2, results_path):

    _, _, good_matches = feature_matching(descriptor1, descriptor2, corners1, corners2, 0.6)
    matches_array = np.array(good_matches)

    image_feature_matches = plot_feature_correspondance(image1, image2, matches_array)
    plt.imsave(results_path + "feature_matches_" + str(i+1) + "_" + str(j+1) + ".png" , image_feature_matches)

    if len(matches_array) > 4:
        
        h_matrix, inlier_matches, ratio_inliers = RANSAC(matches_array, 1000, 20)
        print(ratio_inliers)
        
        if many_to_one_matches_check(inlier_matches):
            return return_no_matches()
        image_filtered_matches_RANSAC = plot_feature_correspondance(image1, image2, inlier_matches)
        plt.imsave(results_path + "feature_matches_RANSAC_" + str(i+1) + "_" + str(j+1) + ".png" , image_filtered_matches_RANSAC)

        return h_matrix, ratio_inliers, len(inlier_matches)  
    else:
        print("Very few matches found. Stitching not recommended!")
        return return_no_matches()

def matching_and_stitching_pipeline(image1, image2, i, results_path, corner_method = "Harris", blending_method = "simple"):

    corners1, descriptor1 = corner_pipeline(image1, i, results_path, corner_method)
    corners2, descriptor2 = corner_pipeline(image2, i+1, results_path, corner_method)
    _, _, good_matches = feature_matching(descriptor1, descriptor2, corners1, corners2, 0.75)
    matches_array = np.array(good_matches)

    # image_feature_matches = plot_feature_correspondance(image1, image2, matches_array)
    # plt.imsave(results_path + "feature_matches_" + str(i+1) + "_" + str(i+2) + ".png" , image_feature_matches)
    if len(matches_array) > 4:
        h_matrix, inlier_matches, ratio_inliers = RANSAC(matches_array, 1000, 100)
        # h_matrix, inlier_matches, ratio_inliers = RANSAC(matches_array, 1000, 5)
        print("Ratio of inliers found after RANSAC for images ", i+1, " and", i+2, " : ", ratio_inliers)

        if many_to_one_matches_check(inlier_matches):
            return return_no_matches()
    else:
        print("Very few matches found. Stitching not recommended!")
        return return_no_matches()

    stitched_image = stitch_the_image(image1, image2, h_matrix)
    plt.imsave(results_path + "feature_matches_warped_" + str(i+1) + "_" + str(i+2) + ".png" , stitched_image)

    return stitched_image, ratio_inliers, len(inlier_matches)

    # result_image = blend_the_image(image_array[i], stitched_image, method = "simple")
    # plt.imsave(results_path + "blended_output_" + str(i+1) + "_" + str(j+1) + ".png" , result_image)

def main():
    image = []
    path = os.path.dirname(os.getcwd()) 

    filepath = path + "\\Data\\Train\\Set1\\"
    # filepath = path + "\\Data\\Test\\TestSet2\\"

    results_path = path + "\\Results\\" + filepath.split("\\")[-3] + "\\" + filepath.split("\\")[-2] + "\\"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    for i in os.listdir(filepath):
        im = cv2.imread(filepath + i)
        image.append(im)
        plt.imsave(results_path + i, im)


    corners = []
    anms_img = []
    descriptors = []

    corner_methods = ["Harris", "Shi-Tomasi"]

    image_new = image[0]

    for i in range(len(image) - 1):
        '''
        Sequential stitching
        '''
        image_new, _, status = matching_and_stitching_pipeline(image_new, image[i+1], i, results_path)
        if not status:
            print("Last stitched image stored, aborting stitch")
            break
    
    if status:
        print("Successful stitching performed!")

if __name__ == "__main__":
    main()