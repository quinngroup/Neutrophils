'''In continue with binarization2 code, here I tried to use the standard blob detection in different aspects, using objects shape, color,
size and other features.
Mojtaba , Jul,07,2016
'''

import cv2, math

import argparse
import scipy as sp
import numpy as np
import scipy.ndimage as nd

from random import randint
import sys
import imutils
import matplotlib.pyplot as plt

# Defining arguments for parsing the input file and buffer size
Argu = argparse.ArgumentParser()
Argu.add_argument("-v", "--video",
	help="path to the (optional) video file")
#Argu.add_argument("-b", "--buffer", type = int, default=48,
#	help="max buffer size")
args = vars(Argu.parse_args())

#bsize = args["buffer"]
video_file = 'Phagocytosis_dmso_control 06302016.avi'

# importingg the video file
vf = cv2.VideoCapture(video_file)

X = int(vf.get(3))
Y = int(vf.get(4))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('123-2.avi',fourcc, 40.0, (X, Y))


#List = npzeros()
print (X, Y)

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('123.avi',fourcc, 20.0, (X, Y))
blob_numbers = []
frame_number = 0
# check if the file is opened the do the processes
#alpha = 1/3.0
alpha2 = 1/9.5
#gamma_table = np.array([((i/255.0) ** alpha) * 255.0 if i > 20 else 0 for i in range(0,256)]).astype(np.uint8)
gamma_table2 = np.array([((i/255.0) ** alpha2) * 255.0 if i > 10 else 0 for i in range(0,256)]).astype(np.uint8)
#colors=[(255,255,randint(0,255)) for x in range(400)]
while vf.isOpened():
   
    _, frame = vf.read()
    original_frame = frame
    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
    BW = np.array([i if i > 106 else 0 for i in range(0,256)]).astype(np.uint8)

    kernel = np.ones((3, 3), "uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    frame = clahe.apply(frame)
    
    frame2 = cv2.LUT(frame,BW)
    frame3 = cv2.LUT(frame2,gamma_table2)

    
    #frame4 = final = cv2.medianBlur(frame3, 3)
    #frame4 = final = cv2.medianBlur(frame3, 3)
    #cv2.fastNlMeansDenoising(frame3,frame3,5,5)
                    #cv2.filter2D(frame3,-1,kernel)
    blur = cv2.bilateralFilter(frame,9,75,75)
    frame4 = final = cv2.medianBlur(blur, 3)


    #frame5= cv2.Laplacian(255-frame4,cv2.CV_64F,ksize= 3)
    #frame5 = cv2.erode(frame4, (3,3),iterations=7)
    frame5= cv2.GaussianBlur(frame4,(7,7),0)
    
    frame5= cv2.Laplacian(frame5,cv2.CV_8S,ksize= 5)
    frame5= frame5.astype(np.uint8)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 2000

    params.filterByColor = False
    params.blobColor > 0


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 350
    params.maxArea = 1300
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(frame4)
    blob_numbers.append((frame_number, len(keypoints)))
    frame_number+= 1
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    #print("here")
    print(dir(keypoints[0]))
    #print(keypoints[0].pt)  #point center
    #import sys
    
    im_with_keypoints = cv2.drawKeypoints(frame4, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keypoint in keypoints:
        cur_center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
        im_with_keypoints= cv2.putText(im_with_keypoints, "Here", cur_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)
    #sys.exit(0)
    #BW2 = np.array([32767 if i == 32767 else 0 for i in range(0,256)]).astype(np.int16)
    #frame5 = cv2.LUT(frame5,BW2)   
    #frame5 = cv2.inRange(frame5,32767,32767)
    #print(cv2.minMaxLoc(frame5))
    #from scipy.signal import argrelmax, argrelmin
    ##x,y = argrelmax(frame5)
    #for i in zip(y,x):
    #    frame5 = cv2.circle(frame5,i,3,1000,-1)
    #    #print(i)
    #kernel = np.ones((3, 3), "uint8")
    #frame3 = cv2.erode(frame3, kernel,iterations=1)
    #frame3 = cv2.dilate(frame3, kernel,iterations=1)

    #sobelx = cv2.Sobel(florescent2, cv2.CV_64F, 1, 0, ksize = 5)
    #sobely = cv2.Sobel(florescent2, cv2.CV_64F, 0, 1, ksize = 5)
 
    #edges = cv2.Canny(florescent2, 20, 250)
    #edges = cv2.erode(edges, kernel,iterations=1)
    #edges = cv2.dilate(edges, kernel,iterations=1)

 #****************************{ laplacian gaussian }*************************************************************************************
 #   lena = sp.misc.ascent()
 #   LoG = nd.gaussian_laplace(lena, 2)
 #   thres = np.absolute(LoG).mean() * 0.75
 #   output = sp.zeros(LoG.shape)
 #   w = output.shape[1]
 #   h = output.shape[0]

 #   for y in range(1, h - 1):
 #       for x in range(1, w - 1):
 #           patch = LoG[y-1:y+2, x-1:x+2]
 #           p = LoG[y, x]
 #           maxP = patch.max()
 #           minP = patch.min()
 #           if (p > 0):
 #               zeroCross = True if minP < 0 else False
 #           else:
 #               zeroCross = True if maxP > 0 else False
 #           if ((maxP - minP) > thres) and zeroCross:
 #               output[y, x] = 1
#{***********************************************************************************************************************}

    cv2.imshow("Keypoints", im_with_keypoints)
    #print(im_with_keypoints.dtype)
   # cv2.imshow('Blur',blur)
   # cv2.imshow('median',frame4)
   # cv2.imshow('lap',frame5)
 #   cv2.imshow('lap_gau',output)
    
    #cv2.imshow('BLUR',frame3)
   
    #cv2.imshow('original',original_frame)
    #for x in range(0,1024):
    #    for y in range(0,1024):
    #        florescent2[y,x] = 0 if edges[y,x] >0 else florescent2[y,x]
   
    out.write(im_with_keypoints) #cv2.cvtColor(im_with_keypoints, cv2.COLOR_GRAY2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
'''#zip(*blob_numbers)
plt.scatter(*zip(*blob_numbers))
plt.plot(*zip(*blob_numbers))
plt.show()
print blob_numbers
'''
cv2.destroyAllWindows()
out.release()
vf.release()