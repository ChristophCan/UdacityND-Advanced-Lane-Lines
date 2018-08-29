# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:54:40 2018

@author: admin
"""

import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
 
    
def calibrateFromChessboard():
    
    #Generate ObjectPoints
    objp = np.zeros((9*6, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        
    #Prepare List of ObjectPoints and ImagePoints of calibration-Images
    objectPoints = []
    imagePoints =  []
    
    #Loop over all calibration images
    for i, imagePath in enumerate(glob.glob('camera_cal\*.jpg')):
        
        print("Calibrating from image " + imagePath)
        
        #Load image and convert to gray
        image = mpimg.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #Enhancing the gray image to get better contrast for finding corners
        gray = np.uint8(255 * (gray/np.max(gray)))
                    
        #Get Image-Points by detecting chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        #If chessboard corners found, append to lists
        if ret==True:
            objectPoints.append(objp)
            imagePoints.append(corners)
            
            # Draw and save the corners
            cv2.drawChessboardCorners(image, (9,6), corners, ret)
            write_name = 'output_images\chessboardCorners\corners_found'+str(i)+'.jpg'
            cv2.imwrite(write_name, image)                        
            
    #Read one calibration image as test image for undistorting
    index_image = 1
    test_undist_image = mpimg.imread('camera_cal\calibration'+str(index_image)+'.jpg')
    img_size = (test_undist_image.shape[1], test_undist_image.shape[0])
    
    #Calculate parameters for undistorting
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, img_size,None,None)
    
    #Undistort image and save result
    dst = cv2.undistort(test_undist_image, mtx, dist, None, mtx)
    undist_write_name = 'output_images\chessboardCorners\calibration'+str(index_image)+'_undistort.jpg'
    cv2.imwrite(undist_write_name, dst)
    
    #Write parameters to pickle for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )
    
print("Starting calibration...")
calibrateFromChessboard()
print("Calibration finished!")