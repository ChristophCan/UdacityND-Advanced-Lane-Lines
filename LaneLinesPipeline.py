# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 20:12:09 2018

@author: admin
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as pyplot
import numpy as np
import cv2
import pickle
from line import Line
from moviepy.editor import VideoFileClip

#Save images of each pipeline step as jpg
image_output = False

#Initiale Lines
left_Lane = Line()
right_Lane = Line()

Minv = None

#Undistort the image
def imageUndistort(image):
    # Read in the pickle saved mtx and dist parameters
    dist_pickle = pickle.load( open( "camera_cal\wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
        
    #Undistort image
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    
    #save undistored image for documentation
    if image_output:        
        saveImage = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./output_images/01_Undistort.jpg', saveImage)
        
    return dst

#Helper Functions for creating binary image####################################

#Applies a Sobel to the img 
def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh = (0, 255)):    
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel_img = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize = sobel_kernel)    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel_img)    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale_sobel = np.uint8(255*(abs_sobel/np.max(abs_sobel)))    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scale_sobel)
    binary_output[(thresh[0]<scale_sobel) & (thresh[1]>scale_sobel)] = 1        
    # 6) Return this mask as your binary_output image
    #binary_output = sobel_img # Remove this line
    
    if image_output:
        cv2.imwrite('./output_images/03_Sobel' + ('X' if orient=='x' else 'Y') + '.jpg', binary_output*255)
    
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    # 2) Take the gradient in x and y separately
    xsobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    ysobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)    
    # 3) Calculate the magnitude 
    sobel_magnitude = np.sqrt(xsobel**2 + ysobel**2)    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_magnitude = np.uint8(255*(sobel_magnitude/np.max(sobel_magnitude)))    
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scale_magnitude)
    binary_output[(mag_thresh[0]<scale_magnitude) & (mag_thresh[1]>scale_magnitude)] = 1    
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    
    if image_output:
        cv2.imwrite('./output_images/04_MagThreshold.jpg', binary_output*255)
    
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)    
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(thresh[0]<dir_sobel) & (thresh[1]>dir_sobel)] = 1    
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    
    if image_output:
        cv2.imwrite('./output_images/05_DirThreshold.jpg', binary_output*255)
    
    return binary_output

#Threshold in saturation channel of image
def hls_s_threshold(img, s_thresh=(0,255)):
    
    #Convert to HLS
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)    
    s_image = hls_image[:,:,2]    

    #Create S-Binary Image
    binary_output = np.zeros_like(s_image)
    binary_output[(s_image > s_thresh[0]) & (s_image < s_thresh[1])] = 1
    
    if image_output:
        cv2.imwrite('./output_images/02_SatThreshold.jpg', binary_output*255)
    
    return binary_output

#Helper Functions for finding lane pixels and fitting polynoms#################

def find_lane_pixels(binary_warped):
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current+margin  # Update this
        win_xright_low = rightx_current-margin  # Update this
        win_xright_high = rightx_current+margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        #good_left_inds = binary_warped[win_y_low:win_y_high,win_xleft_low:win_xleft_high].nonzero()[1]
        #good_right_inds = binary_warped[win_y_low:win_y_high,win_xright_low:win_xright_high].nonzero()[1]
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        amount_pixels_left = len(good_left_inds)
        if amount_pixels_left>minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        
        amount_pixels_right = len(good_right_inds)        
        if amount_pixels_right>minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        #pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    
    #set lane pixel values for left and right line
    left_Lane.allx = leftx
    left_Lane.ally = lefty
    right_Lane.allx = rightx
    right_Lane.ally = righty  
            
    ### Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, deg=2)
    right_fit = np.polyfit(righty, rightx, deg=2)

    #set lane current fit values for left and right line
    left_Lane.current_fit = left_fit
    right_Lane.current_fit = right_fit    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]        
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
   
    # Plots the left and right polynomials on the lane lines
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])    
    cv2.polylines(out_img, np.int_([pts_left]), False, (0,255,255), 2)
    cv2.polylines(out_img, np.int_([pts_right]), False, (0,255,255), 2)    

    if image_output:
        cv2.imwrite('./output_images/09_fit_SlidingWindow.jpg', out_img)
    
    return left_fitx, right_fitx

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, deg=2)
    right_fit = np.polyfit(righty, rightx, deg=2)
    
    #set lane current fit values for left and right line
    left_Lane.current_fit = left_fit
    right_Lane.current_fit = right_fit  
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###        
    left_fit = left_Lane.best_fit
    right_fit = right_Lane.best_fit
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                      (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin))).nonzero()[0]
    
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    #set lane pixel values for left and right line
    left_Lane.allx = leftx
    left_Lane.ally = lefty
    right_Lane.allx = rightx
    right_Lane.ally = righty

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])    
    cv2.polylines(result, np.int_([pts_left]), False, (0,255,255), 2)
    cv2.polylines(result, np.int_([pts_right]), False, (0,255,255), 2)

    ## End visualization steps ##
    
    if image_output:
        cv2.imwrite('./output_images/10_fit_SearchAround.jpg', result)
    
    return left_fitx, right_fitx

def measure_curvature_real(lane_class, img):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3/70 # meters per pixel in y dimension - one dashed line (3m) is roughly 70px long
    xm_per_pix = 3.7/615 # meters per pixel in x dimension -> The lane lines (3.7m) are roughly 610 px apart
    
    #Get last fit and calculate the y-values and x-values in pixel-dimensions
    current_fit = lane_class.current_fit
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    fitx = current_fit[0]*ploty**2 + current_fit[1]*ploty + current_fit[2]       
        
     #Get Position of vehicle relative to the line in Pixel space
    pos_vehicle = abs((img.shape[1]/2) - fitx[img.shape[0]-1])
    
    #Transform y- and x-values, and position of vehicle into real space
    ploty = ploty * ym_per_pix
    fitx = fitx * xm_per_pix
    pos_vehicle = pos_vehicle * xm_per_pix
    
    #New fit with data from real space
    fit_cr = np.polyfit(ploty, fitx, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    #Implement the calculation of R_curve (radius of curvature) #####
    curverad = ((1+(2*fit_cr[0]*y_eval + fit_cr[1])**2)**(3/2))/np.absolute(2*fit_cr[0])  
    
    #If curverad is higher than 1500m, consider it as straight line
    if curverad>1500:
        curverad = 9999.0
    else:
        curverad = round(curverad, 0)
        
    #Safe curvature and line base position to line object
    lane_class.radius_of_curvature = curverad         
    lane_class.line_base_pos = round(pos_vehicle,2)
           
    return curverad

def drawPolygon(warped,Minv,image,left_Lane,right_Lane):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))    
    
    left_fit = left_Lane.best_fit
    right_fit = right_Lane.best_fit
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    color_warp[left_Lane.ally,left_Lane.allx] = [255,0,0]
    color_warp[right_Lane.ally,right_Lane.allx] = [0,0,255]
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    
    if image_output:
        cv2.imwrite('./output_images/11_drawPolygon.jpg', newwarp)
    
    return newwarp
    
    #figure, ax = pyplot.subplots(1,1, figsize=(35,20))
    #ax.imshow(result)

def resultsMakeSense(left_Lane, right_Lane):
    #Checking that they have similar curvature
    curvature_diff = np.abs(left_Lane.radius_of_curvature - right_Lane.radius_of_curvature)
    if curvature_diff > 1000:
        return False
    
    #Checking that they are separated by approximately the right distance horizontally 
    lane_dist = left_Lane.line_base_pos + right_Lane.line_base_pos
    if (lane_dist > 4) | (lane_dist < 3.4):
        return False
    
    #Checking that they are roughly parallel 
    fit_diff = np.abs(left_Lane.current_fit - right_Lane.current_fit)
    if (fit_diff[0] > 1.0e-03) | (fit_diff[1] > 1.5):
        return False
    
    return True

def process_image(image):
    #Undistort the Image
    undist_image = imageUndistort(image)

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    s_binary = hls_s_threshold(undist_image, s_thresh=(150,255)) #120

    gradx = abs_sobel_thresh(undist_image, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    grady = abs_sobel_thresh(undist_image, orient='y', sobel_kernel=ksize, thresh=(20, 200)) #20

    mag_binary = mag_thresh(undist_image, sobel_kernel=ksize, mag_thresh=(50, 200)) #30
    dir_binary = dir_threshold(undist_image, sobel_kernel=ksize, thresh=(0.7, 1.3)) #0.7 1.3

    #Combining the binary images
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    if image_output:
        cv2.imwrite('./output_images/06_combinedBinary.jpg', combined*255)

    #Warpe the image
    src = np.float32([[590,450],[695,450],[240,680],[1100,680]])
    dst = np.float32([[200,0],[combined.shape[1]-400,0],[200,combined.shape[0]],[combined.shape[1]-400,combined.shape[0]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined, M, (combined.shape[1], combined.shape[0]), flags=cv2.INTER_LINEAR)    
    
    if image_output:
        image_with_lines = np.copy(undist_image)
        cv2.line(image_with_lines, (src[0][0], src[0][1]), (src[1][0], src[1][1]), (255,0,0), 1)
        cv2.line(image_with_lines, (src[1][0], src[1][1]), (src[3][0], src[3][1]), (255,0,0), 1)
        cv2.line(image_with_lines, (src[3][0], src[3][1]), (src[2][0], src[2][1]), (255,0,0), 1)
        cv2.line(image_with_lines, (src[2][0], src[2][1]), (src[0][0], src[0][1]), (255,0,0), 1)
        
        warpedOriginal = cv2.warpPerspective(image_with_lines, M, (image_with_lines.shape[1], image_with_lines.shape[0]), flags=cv2.INTER_LINEAR)
        cv2.imwrite('./output_images/07_warpedImage.jpg', cv2.cvtColor(warpedOriginal, cv2.COLOR_RGB2BGR))        
        
        cv2.imwrite('./output_images/08_warpedImageBinary.jpg', warped*255)

    #Finding line pixels and fitting the polynom
    if left_Lane.detected==False | right_Lane.detected==False:
        #Initial Fit, if no lines have been detected in the previous frame
        left_fitx, right_fitx = fit_polynomial(warped)
    else:
        #Following fit, if lines have been detected in the previous frame
        left_fitx, right_fitx = search_around_poly(warped)
    
    #Calculate the curve radius
    left_curveRad = measure_curvature_real(left_Lane, warped)
    right_curveRad = measure_curvature_real(right_Lane, warped)
    
    #Sanity Check
    if resultsMakeSense(left_Lane, right_Lane):
        #Append xfits to left and right line
        left_Lane.appendRecentXFitted(left_fitx, warped)
        right_Lane.appendRecentXFitted(right_fitx, warped)   
        
        left_Lane.detected = True
        right_Lane.detected = True
    else:
        left_Lane.detected = False
        right_Lane.detected = False
    
        
    #line_img = left_Lane.drawCurrentFit(warped)
    #line_img = right_Lane.drawCurrentFit(line_img)
    #ax10.imshow(line_img)

    newwarp = drawPolygon(warped, Minv, undist_image,left_Lane,right_Lane)
    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)
    #ax10.imshow(result)
    
    #Print Curve Radius to image
    cv2.putText(result, 'LeftCurve: ' + str(left_curveRad), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(result, 'RightCurve: ' + str(right_curveRad), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(result, 'TotalCurve: ' + str((right_curveRad+left_curveRad)/2), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    
    #Print Distance of vehicle to image    
    cv2.putText(result, 'Dist. from left: ' + str(left_Lane.line_base_pos), (850,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(result, 'Dist. from right: ' + str(right_Lane.line_base_pos), (850,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(result, 'Center: ' + str(round(abs((right_Lane.line_base_pos + left_Lane.line_base_pos)/2 - left_Lane.line_base_pos), 2)), 
                (850,130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    
    #Status of Line detection
    cv2.putText(result, 'Lines Detected: ' + str(left_Lane.detected), (10,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    
    """        
    # Plot the result for debugging 
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = pyplot.subplots(5, 2, figsize=(20, 20))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=25)
    
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=25)

    ax3.imshow(gradx, cmap='gray')
    ax3.set_title('Sobel x-direction.', fontsize=25)
    
    ax4.imshow(grady, cmap='gray')
    ax4.set_title('Sobel y-direction.', fontsize=25)
    
    ax5.imshow(mag_binary, cmap='gray')
    ax5.set_title('Sobel Magnitude', fontsize=25)
    
    ax6.imshow(dir_binary, cmap='gray')
    ax6.set_title('Sobel Direction', fontsize=25)
    
    ax7.imshow(s_binary, cmap='gray')
    ax7.set_title('S Binary', fontsize=25)
    
    ax8.imshow(warped, cmap='gray')
    ax8.set_title('Warped Combined', fontsize=25)    
    
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)    
    """
    
    if image_output:
        cv2.imwrite('./output_images/12_finalOutput.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    return result

"""
#Loading a test image
image_url = 'test_images/test2.jpg'
image = mpimg.imread(image_url)

if image_output: 
    saveImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./output_images/00_Original.jpg', saveImage)

#
result1 = process_image(image)
result2 = process_image(image)

fig, ax = pyplot.subplots(1,1, figsize=(30,20))
ax.imshow(result1)

"""
#Video output
white_output = 'output_videos/project_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)