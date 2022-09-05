#Importing necessary libraries
import numpy as np
import argparse
import cv2

# For parsing in arguments in the command line 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-p1", "--pixel1", required = True, help = "Pass pixel values")
ap.add_argument("-p2", "--pixel2", required = True, help = "Pass pixel values")
args = vars(ap.parse_args())

# Reading image
image = cv2.imread(args["image"])
# Reading the pixel dimensions (to check whether it lies inside the circle or not)
xx = args["pixel1"]
yy = args["pixel2"]

# creating a copy of the original image to write point inside or outside and draw over the detected circles
output = image.copy()
#convert the color image to gray scale and then blur it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(gray, (3,3))

# converting the parsed in values to integer
pixel = (int(xx), int(yy))

minDist = 100
param1 = 60 #500
param2 = 50 #200 #smaller value-> more false circles
minRadius = 5
maxRadius = 250 #10

# To detect circles
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles is not None:
	circles = np.round(circles[0, :]).astype("int")

	circles_containing_pixel = []

	for (x, y, r) in circles:
		cv2.circle(output, (x, y), r, (0, 255, 0), 4) # To draw the circle
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1) # To draw a small rectangle to indicate the center of the circle

		if ((pixel[0] - x)^2 + (pixel[1] - y)^2) < (r^2):
			store = (x, y, r)
			circles_containing_pixel.append(store) # storing the circles which contains the pixel -> if the question is to be modified, we can use this on such case
			flag = 1
		else:
			flag = 0

	if flag == 1:
		cv2.putText(output, 'pixel inside!', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
	else:
		cv2.putText(output, 'pixel outside!', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

	cv2.circle(output, pixel, radius=1, color=(0, 255, 0), thickness=3)
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)