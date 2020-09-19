# Advanced Lane Finding

## **Overview**
Lane finding is an essential function that an autonomous car must have to safely navigate streets. As a fundamental component to building autonomous vehicles, lane finding must be highly accurate and robust to variable road surfaces, lighting conditions, and road curvatures. This project explores advanced computer vision techniques to build a more robust lane finding pipeline. These techniques include exploring color spaces other than RGB, camera calibration, and perspective transforms. 

The steps taken to build the lane finding pipeline are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[calibration]: ./examples/calibration.png "Calibration Image"
[top_down]: ./examples/top_down.png "Perspective Transform Image"
[binary]: ./examples/binary.png "Binary Image"
[sliding_window]: ./examples/sliding_window.png "Sliding Window Lane Detection"
[line_region]: ./examples/line_region.png "Line Region Lane Detection"
[output]: ./examples/output.png "Output Image"

## **Methodology**
### 1. Camera Calibration
The first step to accurately labeling lane lines is correcting for camera lens distortion. We can do this by taking multiple pictures of a chessboard from different angles and distances. OpenCV has a built in function for detecting chessboard corners and another function to calibrate the camera based on the detected corners. The below function reads in 19 chessboard images, locates the chessboard corners, calibrates the camera, and returns camera matrix and distortion coefficients needed to undistort an image from this camera. 
```
def calibrate_camera():
    image_fnames = glob.glob('camera_cal/*.jpg')

    nx = 9
    ny = 6
    img_points = []
    obj_points = []

    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for fname in image_fnames:
        img = cv2.imread(fname)

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)

    ret, undist_mtx, undist_dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[1:], None, None)
    return undist_mtx, undist_dist
```

Simply passing the camera matrix and distortion coefficients, in addition to an image, to the `cv2.undistort()` function will return a corrected image. Below is a uncorrected and corrected image for comparison.

![alt text][calibration]

### 2. Perspective Transformation
A key step in building a robust lane line detection pipeline is the perspective transform to generate a top-down view of the road ahead. The top-down view allows the lanes to be represented as parallel to one another. Once parallel, its must easier to fit a polynomial line to each lane line and perform sanity checks that the lane curvature is within reason. The perspective transformation in my pipeline is packaged into a function named `perspective_warp()` as seen below.

```
def perspective_warp(undist, src, dst):
    img_size = (undist.shape[1], undist.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M_inv
```

The function takes in two parameters named `src` and `dst`. These are vertices defining the region of the original image (`src`) to be transformed into a region of the resulting image (`dst`). These vertices have been hard coded with the following values:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 300, 720      | 
| 589, 450      | 300, 0        |
| 690, 450      | 980, 0        |
| 1130, 720     | 980, 720      |

Performing the perspective transform on a test image produces the top-down image seen below.

![alt text][top_down]

### 3. Binary Thresholding
The next step in detecting and identifying lane lines is converting the color image to a binary image showing only, in an ideal world, the lane line pixels. The conversion from color to binary image is the most open ended component of the pipeline because of the wide variety of image features that can be used for thresholding. These features include gradient magnitudes and direction and a variety of color spaces (RGB, HSV, and HLS), all of which can have a custom threshold value.

The color to binary function implemented in this pipeline can be seen in the code snippet below. First the image is converted to the HLS color space. The sobel operator for the x direction is applied to the L channel to locate gradients in the x direction. The sobel-x image is then thresholded to knock out noise in the image. Separately the S channel is thresholded to highlight the lane line features. Finally, the x gradient thresholded image and S channel thresholded image are combined into a final binary image.

```
def binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary
```

Below is an example of the `binary()` function applied to a sample image.

![alt text][binary]

### 4. Polynomial Fit
The binary image is fed to a lane line finding and polynomial fitting function to perform the final lane line identification. The binary pixels are interpreted as lane lines by studying the left and right sides of the image separately to find the most likely lane line feature on either side. The bottom half of the image is then analyzed by summing the pixel values in each column of the image to generate a histogram of pixel values across the x-axis. The peaks on the left and right sides of the image are used as the starting point for finding pixels using the sliding window technique. 

The sliding window technique works by placing two small windows along the bottom of the image on the left and right histogram peaks. Next new windows are stacked on top of the previous windows and then are shifted left or right to center them around the highest concentration of pixels in the window. More windows are iteratively stacked on top of each other until the top of the image is reached. Finally, a second order polynomial is fit to the pixels found within the sliding windows.

The pixel finding and polynomial fitting functions can be found in the `fit_polynomial()` function defined in the `Advanced-Lane-Lines-Pipeline` notebook. When this pipeline is applied to a video, the coefficients found by fitting the left and right lines in the first image of the video are then used to define a search region around the fit line in the next image of the video. Searching within a region around the fit line reduces the time spent finding pixels in the next image. A minimum pixel threshold is used to determine is the pipeline should fall back to the sliding window method of finding pixels. 

Below is an example of the lane line pixels found using the sliding window function and a polynomial lines fit to the pixels.

![alt text][sliding_window]

After the sliding window fits the initial line, the regional pixel finding method of searching next to the previous line kicks in as demonstrated in the following image.

![alt text][line_region]

### 5. Lane Curvature
An important step in applying computer vision to lane line detection is performing sanity checks on the results of the pipeline. A sanity check is a verification on the detected lane lines to check if the line appears reasonable. For example, a reasonable radius of curvature of the lane, which is anywhere from 500 m to 10 km for highways. Radius of curvature, in meters, can be calculated from the coefficients of a second order polynomial using the function defined in `measure_curvature_real()`. This function accepts the left and right points that have been identified as lane line points, which are then fit to a second order polynomial, and the coefficients are passed to the curvature calculation.

```
def measure_curvature_real(leftx, rightx, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/1280 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad
```

After identifying the lane lines, the vehicle's deviation from the center of the lane can be calculated by observing the distance of the two lane lines from the mid-point of the image. This calculation can be considered reasonably accurate for determining deviation from center because we can assume that the camera is mounted at the center of the vehicle and pointing straight forward.

### 6. Project Lane Lines onto Road
Finally we can visualize the lane lane finding pipeline by projecting the polynomial fit lines onto the original input image. The projection process is the inverse of the first perspective transform to create the top-down image. Once we have lane lines drawn on a blank, top-down image, we can perform a transform to return to the original perspective and layer the lane lines onto the original image. See below for the final output of the lane finding pipeline.

![alt text][output]

## **Pipeline**
The lane finding pipeline can be applied to a video to demonstrate how it would perform in a live autonomous vehicle. The link below is a video output from the lane finding pipeline with lane lanes superimposed on the images. It can be seen that this pipeline performs reasonably well on a relatively simple highway driving example.

[Pipeline_Video](test_videos/project_output.mp4)

## **Discussion**
This pipeline is far from a complete lane detection system as it will begin to break down on tightly turning roads or roads with overpasses and tunnels. Below are a few thoughts for improvements to the pipeline to increase accuracy and robustness.

1. Further refinement of color and gradient thresholding. This pipeline will struggle to find lane lines when very dark and large shadows are cast on the road, such as overpass. Better color and gradient thresholding would help isolate lane lines in these situations. Similarly, mixed road surfaces like the adjacent concrete and asphalt road surfaces also become a challenge for this pipeline. Better selecting for the white and yellow lane lines would help push concrete-asphalt edges to the background.

2. Improved handling of roads with small turning radius'. This pipeline achieves some of its stability by applying a sanity check that the turning radius of the current lane line has a radius of curvature greater than 500m. If the lane line has a radius of less then or equal to 500m, the current fit line is ignored and the pipeline falls back to the previous fit line. More dynamic sanity checks that allow for a smaller radius of curvature would make the pipeline robust to more roads.
