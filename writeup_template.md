## Writeup / README Christoph Chalfin

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chessboardCorners]: ./output_images/chessboardCorners/corners_found13.jpg "ChessboardCorners"
[undistortedImage]: ./output_images/chessboardCorners/calibration1_undistort.jpg "UndistortedImage"
[distortedImage]: ./camera_cal/calibration1.jpg "DistortedImage"
[distortedImage2]: ./output_images/00_Original.jpg "distortedImage2"
[undistortedImage2]: ./output_images/01_Undistort.jpg "undistortedImage2"
[SatThreshold]: ./output_images/02_SatThreshold.jpg "SatThreshold"
[SobelX]: ./output_images/03_SobelX.jpg "SobelX"
[SobelY]: ./output_images/03_SobelY.jpg "SobelY"
[MagThreshold]: ./output_images/04_MagThreshold.jpg "MagThreshold"
[DirThreshold]: ./output_images/05_DirThreshold.jpg "DirThreshold"
[combinedBinary]: ./output_images/06_combinedBinary.jpg "combinedBinary"
[warpedImage]: ./output_images/07_warpedImage.jpg "warpedImage"
[warpedImageBinary]: ./output_images/08_warpedImageBinary.jpg "warpedImageBinary"
[slidingWindow]: ./output_images/09_fit_SlidingWindow.jpg "slidingWindow"
[previousFit]: ./output_images/10_fit_SearchAround.jpg "prevoiusFit"
[polygon]: ./output_images/11_drawPolygon.jpg "polygon"
[finaloutput]: ./output_images/12_finalOutput.jpg "finaloutput"

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Calibration Pipeline with an example

The Camera Calibration is done in the python file `CameraCalibration.py` with the method `calibrateFromChessboard()`.

The method first generates the **object points** in 3D space on the checkboard. I am using numpy´s `mgrid`, in combination with the `T` (Transpose) and `reshape` method. I have to admit, that I just copy pasted this from the class code. I haven´t fully understood yet, how the `mgrid` method in general is useful for other problem solvings.

Next I am looping through all checkboard images. Each image first is converted to gray colorspace to feed it into the `findChessboardCorners` method. As it already says in it´s name, it is looking for the corners of the chessboard image. These corners then serve as my **image points**. I also implemented an enhancement to the image for using the full range of 255 grayscale values, in order to get better results on detecting the chessboard corners:

    gray = np.uint8(255 * (gray/np.max(gray)))

The following picture shows an example:

![ChessboardCorners][ChessboardCorners]

After that the **object points** and **image points** are used in the method `calibrateCamera` for computing the parameters I need for undistorting my images. The most important parameters are the cameraMatrix and the distCoeffs. The following pictures show an example of a distorted und undistorted image:

![DistortedImage][DistortedImage]
![UndistortedImage][UndistortedImage]

The parameters are finally stored in a pickle-file for later use.

### Pipeline (single images)

#### 1. Distortion-corrected image.

The first step of the pipeline is to undistort the camera image. As an example, I am using the image `test_images/test2.jpg`

![distortedImage2][distortedImage2]

The undistorting is done with the method `imageUndistort(image)`, which takes the distorted image as an argument and returns the corrected version. The therefore necessary parameters are saved in the pickle file `camera_cal\wide_dist_pickle.p`, this was done by the CameraCalibration routine as described above.

![undistortedImage2][undistortedImage2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used three general threshold methods to generate a final binary result image. All methods take the undistorted image as input.

1. Threshold in the saturation channel of the image `hls_s_threshold(img, s_thresh)` _lines 118 to 131_
2. Sobel Operator in x- and y-direction `abs_sobel_thresh()` _lines 44 to 64_
3. Threshold in magnitude and direction of ... `mag_thresh()` and `dir_threshold()` _lines 69 to 115_

The three outputs are then combined into one binary output image. The above methods and the combining is called in _lines 467 to 478_.

The following pictures show the output images of each step.

Saturation Threshold
![SatThreshold][SatThreshold]
Sobel in x-direction
![SobelX][SobelX]
Sobel in y-direction
![SobelY][SobelY]
Magnitude Threshold
![MagThreshold][MagThreshold]
Direction Threshold
![DirThreshold][DirThreshold]
Combined binary
![combinedBinary][combinedBinary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transormation is performed in _lines 483 to 489_. I first chose source and destination points for the transformation. The `combined` is the combined_binary image from above.

```python
src = np.float32(
    [[590,450],
    [695,450],
    [240,680],
    [1100,680]])
dst = np.float32(
    [[200,0],
    [combined.shape[1]-400,0],
    [200,combined.shape[0]],
    [combined.shape[1]-400,combined.shape[0]]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Warped color image:
![warpedImage][warpedImage]
Warped binary image:
![warpedImageBinary][warpedImageBinary]

### 4. Usage of the Line class

As mentioned in the Tips and Tricks for the project, I defined a `Line()` class with the fields mentioned. I first tried to understand, what each of the fields meant for my pipeline.

---

    # was the line detected in the last iteration?
    self.detected = False

In order to keep track if my pipeline detected lines that made sense during the last iteration, this field is either set to `True` or `False`. If it is true, I can use the data from this iteration as my starting point for the current one with `search_around_poly(warped)`. If it is false, I have to completely start over and use a initial histogramm and sliding window method with `fit_polynomial(warped)`.

---

    # x values of the last n fits of the line
    self.recent_xfitted = []

A list of x-values, which were confidently fitted during the last n frames. This list is used for calculating the bestx values.

---

    #average x values of the fitted line over the last n iterations
    self.bestx = None    

A mean value over the `recent_xfitted` list.

---

    #polynomial coefficients averaged over the last n iterations
    self.best_fit = None  

The fit over the `bestx`-values. This fit is used for drawing the polygon onto the original footage.

---

    #polynomial coefficients for the most recent fit
    self.current_fit = [np.array([False])]  

Coefficients of the fiting polynomial to the line in the last iteration.

---

    #radius of curvature of the line in some units
    self.radius_of_curvature = None

As mentioned, current radius of the line.

---

    #distance in meters of vehicle center from the line
    self.line_base_pos = None

As mentioned, current distance in meters of the vehicle center from the line

---

    #difference in fit coefficients between last and new fits
    self.diffs = np.array([0,0,0], dtype='float')

I honestly haven´t figured out the use of this parameter. Maybe you can give me a hint?

---

    #x values for detected line pixels
    self.allx = None

All x-values of the pixels, which have been identified to belonging to the lane line.

---

    #y values for detected line pixels
    self.ally = None  

All y-values of the pixels, which have been identified to belonging to the lane line.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

As a first step I am reading the `detected` field from my `left_Lane` and `right_Lane` objects. I then can decide, if I need to totally start over for detecting my lane pixels, or if I can use data from the previuos detection.

In the first case, I am calling the `fit_polynomial`-method, in the latter I can use the `search_around_poly`-method.

---

`fit_polynomial` _lines 135-266_

This method first calls the `find_lane_pixels`-method, which uses the sliding window algorithm to search for binary pixels, that belong either to the left or the right lane. With this information, the `fit_polynomial`-method then fits a polynom at these pixels. The resulting image is shown below:

![slidingWindow][slidingWindow]

---

`search_around_poly` _lines 268-358_

This method uses the fit of the previous frame and searches for the new lane pixels around this previous fit. The resulting image is shown below:

![previousFit][previousFit]

---

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

With the polynomial fits from above, the curvature now can be calculated. This is done by calling the method `measure_curvature_real` _lines 360-401_.

I had to adjust the `ym_per_pix` and `xm_per_pix` according to the warping coordinates I chose. This lead to the following vaules:

- `ym_per_pix = 3/70` (One dashed line, which is 3m in real space, measures 70 px in pixel space)
- `xm_per_pix = 3.7/615` (The distance between left and right lane, which is 3.7m in real space, measures 615 px in pixel space)

With the new fit `fit_cr`, with coordinates transformed to real space, I then calculated the radius of curvature according to the following equation:

Rcurve​=(1+(2Ay+B)^2)^3/2 / |2A|

```python
curverad = ((1+(2*fit_cr[0]*y_eval + fit_cr[1])**2)**(3/2))/np.absolute(2*fit_cr[0])
```

I also considered radius of curvature, which is higher than 1500m, as straight lines. I needed to do this, to get better results for the straight lines. The value of 1500m is arbitrary and might have to be optimized. ​

#### 6. Check if results make sense

Before plotting the results back on the original image, I am checking the following points to make sure, the tracking of the current frame makes sense. This is done by `resultsMakeSense` _lines 442 to 458_

1. Checking that they have similar curvature
2. Checking that they are separated by approximately the right distance horizontally
3. Checking that they are roughly parallel

If one of these checks fails, the current frame is neglected and the next frame will be detected completely new.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The results are plotted back on the original image with the method `draw_Polygon`, which returns the following image:

![polygon][polygon]

This image is then mixed with the original, undistorted image and the information about radius of curvature, distance of vehicle from center of the lane and if the current frame is tracked or not is printed on the image.

![finaloutput][finaloutput]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My biggest problem is that the radius of curvature, which is calculated during the first curve, differs a lot from the expected 1.000m. I can´t find the problem in my pipeline. I tried to recalculate the pixel-space to real-space conversion parameters `ym_per_pix` and `xm_per_pix`, but they looked reasonable to me. I hope you can have a look and maybe give me a hint to the problem.

As the distance of the left and right line always is around 3.7m I am pretty confident, my `xm_per_pix` value is correct.

I am also not very confident about my checking, if the left and right line are roughly parallel. I just roughtly estimated the allowed difference in the A and B parameters of the fit. I hope it will do the job!

I also haven´t tried my pipeline against the `challenge_video.mp4` and `harder_challenge_video.mp3`, but I am expecting some problems when there is change in the color of the street and more or less shady spots.

Another problem occurs on the straight lines. I very often get a radius of curvature around 1.500m or slightly bigger, where it should be much bigger. Maybe this also occurs, because of the picture enhancement pipeline, producing the binary image. There definetly could be done some improvment.  
