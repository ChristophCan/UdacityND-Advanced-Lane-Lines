# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:52:08 2018

@author: admin
"""

import numpy as np
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #Amount of items to save in lists
        self.n=5
        
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
    
    def appendRecentXFitted(self, fitx, img):
        #Append current fitx to the recent_xfitted list
        self.recent_xfitted.append(fitx)
                
        #Delete first entry of recent_xfitted list, if amount of entries is larger than n
        if len(self.recent_xfitted)>self.n:
            self.recent_xfitted.pop(0)        
            
        #Calculate the mean x-values
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        
        #Calculate the polynomial coeffizients of bestx
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        self.best_fit = np.polyfit(ploty,self.bestx,2)
            
        return 0    
    
    
    def drawCurrentFit(self, img):                              
              
        if len(img.shape)>2:
            out_img = img
        else:
            norm_img = 255*(img/np.abs(np.max(img)))
            out_img = np.dstack((norm_img, norm_img, norm_img))
                                
        y1 = 0
        y2 = img.shape[0]-1
        x1 = int(self.current_fit[0]*y1**2 + self.current_fit[1]*y1 + self.current_fit[2])
        x2 = int(self.current_fit[0]*y2**2 + self.current_fit[1]*y2 + self.current_fit[2])
        
        cv2.line(out_img, (x1,y1), (x2,y2), [255,0,0], 3)
                     
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        fit = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]    
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts = np.array([np.transpose(np.vstack([fit, ploty]))])
        #pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        #pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.polylines(out_img, np.int_([pts]), (0,255, 0))
        
        return out_img
        