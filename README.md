# object-detection-and-circle-detection

## Circle Detection

1. Firstly, important libraries are imported
   - numpy : to do certain computations
   - argparse : to parse in arguments in command line
   - cv2 : OpenCV
2. Used arguments:
   - -i or --image : for parsing in image location
   - -p1 or --pixel1 : for parsing x value of pixel
   - -p2 or --pixel2 : for parsing x value of pixel
3. Read image using cv2.imread and x value and y value of pixel
4. Create a copy of the original image using .copy()
5. Convert the colored image to grayscale and then blur it
6. Convert the pixel coordinates into a tuple ( integer values )
7. Find the best values of cv2.HoughCircles by fine tuning (trial and error)
8. Detect the circles using cv2.HoughCircles
9. If circle is not None:
   - round the values of array using np.round
   - initialise a list called “circles_containing_pixel” ( in case we want to explicitly say the circle in
which the pixel is present )
   - loop over all the detected circles
     - Draw the circles using cv2.circle
     - Draw a small rectange to indicate the center of the circle using cv2.rectangle
     - if pixel dimension satisfy the condition → '((pixel[0] - x)^2 + (pixel[1] - y)^2) < (r^2)'
       - flag = 1, meaning the pixel is inside the detected circle
       - else flag=0, meaning the pixel is outside the detected circle
   - if flag == 1 → pixel inside!
   - else→ pixel outside!
10. Display the output image along with the original image for comparison

### How to execute the program in command line:
python answer.py --image {image location} --pixel1 {pixel x value} --pixel2 {pixel y value}
example: python answer.py --image circle.jpg --pixel1 50 --pixel2 50, means the pixel coordinates are (50,50).

## Object Detection

[Untitled.webm](https://user-images.githubusercontent.com/40202626/188384362-4ad71178-c462-4041-a7a8-d31f6ff2487c.webm)

1. Firstly, important libraries are imported
   - numpy
   - argparse
   - time
   - cv2
   - os
2. Used arguments (command line)
   - -i or –input : specify input video location
   - -o or –output : where to create the output file
   - -y or –yolo : yolo model and weights location
   - -c or –confidence : default value = 0.5 (minimum probability to filter weak detections)
   - -t or –threshold : default value = 0.3 (threshold when applying non-maxima suppression)
3. Load COCO class labels using os
4. Specify and dervie the paths to YOLO weights and model
5. Load YOLO object detector (pre trained)
6. Initialize the video stream, pointer to output video → loop over the frames → grab frame by frame
7. If not grabbed, means that the video has ended and if empty, start grabbing frames
8. Construct a blob using the input frame and perform a forward pass of the YOLO object detector
9. Initialize three list to contain the information of
   - boxes
   - confidences
   - classIDs
10. Loop over each Layer outputs (which we got from forward pass → layerOutputs = net.forward(ln))
    - loop over each detections
      - extract classID (label) and confidence (probability)
      - filter out weak predictions
      - if confidence is greater than the given confidence ( default = 0.5 )
        - scale the bounding box coordinates realtive to the size of the object
        - use the center coordinates to derive the top and left corner of the bounding box
        - update the respective lists
11. Apply Non-Maxima Suppression to supress weak, overlapping bounding boxes using cv2.dnn.NMSBoxes
12. Ensure atleast one detection exist → loop over the kept indexes (after NMS)
    - extract bounding box coordinates
    - if confidence is greater than 0.8
      - then color = green `rgb(0,255,0)`
    - else-
      - then color = red `rgb(255,0,0)`
    - draw bounding box rectangle and write label on frame using cv2.rectangle and cv2.putText
13. Check if the video write is None → initialize the video writer → write output frame to disk
14. Finally, release all pointers and write a message “cleanining up” or any message to identify that the process has finished.
