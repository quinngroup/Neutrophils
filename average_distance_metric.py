import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb


cap = cv2.VideoCapture('Phagocytosis_dmso_control 06302016.avi')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

X = int(cap.get(3))
Y = int(cap.get(4))
#List = npzeros()
print (X, Y)
blob_numbers = []
frame_number = 0
avg_Dict = dict()
obj_num = 0
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('eQUALIZED Bilateral.avi',fourcc, 20.0, (X, Y))

alpha2 = 1/4
#gamma_table = np.array([((i/255.0) ** alpha) * 255.0 if i > 20 else 0 for i in range(0,256)]).astype(np.uint8)
gamma_table2 = np.array([((i/255.0) ** alpha2) * 255.0 if i > 1 else 0 for i in range(0,256)]).astype(np.uint8)
obj_dict = dict()
object_num = 0
dist_thresh = 50
frame_thresh = 40
while(1):
    ret, frame1 = cap.read()
    frame_number += 1
    
    frame2 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    BW = np.array([255 if i > 106 else 0 for i in range(0,256)]).astype(np.uint8)

    kernel = np.ones((3, 3), "uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    frame2 = clahe.apply(frame2)
    
    equ = cv2.equalizeHist(frame2)

    bilateral_filtered_image = cv2.bilateralFilter(frame2, 5, 175, 175)
    cv2.imshow('Bilateral', bilateral_filtered_image)
    
    #equ = cv2.equalizeHist(bilateral_filtered_image)
    
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 60, 200)
    cv2.imshow('edge', edge_detected_image)

#***********************************************************************************************************************************
    # apply threshold
    thresh = threshold_otsu(bilateral_filtered_image)
    bw = closing(equ > thresh, square(1))

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=bilateral_filtered_image)

#    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):

        # skip small images
        if region.area < 100 and region.eccentricity > 0.4:
            continue

        # draw rectangle around segmented coins
#        minr, minc, maxr, maxc = region.bbox
#        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                  fill=False, edgecolor='red', linewidth=2)
#        ax.add_patch(rect)
    image_label_overlay_converted =(image_label_overlay*256).astype(np.uint8)
    cv2.imshow('a',image_label_overlay_converted)

#*******************************{ BloB DeTeCtion }*******************************************************************************************    

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 40
    params.maxThreshold = 3000

    params.filterByColor = False
    params.blobColor > 0


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 350
    params.maxArea = 5000
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
  
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)

    #print(dir(keypoints[0]))
    #print(keypoints.class_id)  #point center
    #import sys
    
#********************************************
    # Detect blobs.
    keypoints = detector.detect(bilateral_filtered_image)
    
    # Finding blob centers and creating the path of object during the frames and saving them in a dictionary
    blob_numbers.append((frame_number, len(keypoints)))
    im_with_keypoints = cv2.drawKeypoints(bilateral_filtered_image, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Creating a null dictionary to check if the closest objects are not in a same frame 
    new_objDict = dict()
    lockedObjNums = set()
    for keypoint in keypoints:
        cur_center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
        
       # print (cur_center, frame_number)
        cur_objectNum = -1
        matched = False
        for obj_num in obj_dict :
            obj_occurences = obj_dict[obj_num]

            if ((frame_number - obj_occurences[-1][1]) <= frame_thresh) and (np.sqrt((obj_occurences[-1][0][0]-cur_center[0])**2 + ((obj_occurences[-1][0][1]-cur_center[1])**2)) <= dist_thresh) and obj_num not in lockedObjNums:

                obj_occurences.append((cur_center, frame_number))
                matched = True
                cur_objectNum = obj_num
                lockedObjNums.add(obj_num)
                break
        if not matched : 
            new_objDict[object_num] = [(cur_center, frame_number)]
            cur_objectNum = object_num
            object_num += 1
        im_with_keypoints= cv2.putText(im_with_keypoints, "#"+str(cur_objectNum), cur_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)
    obj_dict.update(new_objDict)
#*******************************************

    #cv2.imshow('Horizontal Component', horz)
    cv2.imshow('Vertical Component', im_with_keypoints)

#**************************************************************************************************************************
    #cv2.imshow('RGB',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next
    #cv2.Cvtcolor(edge_detected_image, COLOR_GRAY2BGR)
  #  outLable = image_label_overlay.astype(np.float32)
    
    out.write(image_label_overlay_converted)

cap.release()
cv2.destroyAllWindows()
#print(obj_dict)

for obj_num in obj_dict : 
    occs = obj_dict[obj_num]
    n = 1
    avg = 0
    frame_Start = 1
    frame_End = 30
    started = False
    Prev_occurence = None
    for occ in occs : 
        if not started :
            if occ[1] < frame_Start : 
                continue
            else : 
                started = True
                Prev_occurence = occ
        else :
            if occ[1] <= frame_End : 
                avg += np.sqrt(((Prev_occurence[0][0]) - occ[0][0])**2 + ((Prev_occurence[0][1]) - occ[0][1])**2)
                n += 1
                Prev_occurence = occ
            else : 
                break
    avg /= float(n)
    avg_Dict[obj_num] = avg
print (avg_Dict)


#**********************************************{ Drawing the Average Distance}*****************************
D = avg_Dict
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())

plt.show()


