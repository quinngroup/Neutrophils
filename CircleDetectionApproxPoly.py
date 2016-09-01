import cv2
import numpy as np

cap = cv2.VideoCapture('Phagocytosis_dmso_control 06302016.avi')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

X = int(cap.get(3))
Y = int(cap.get(4))
#List = npzeros()
print (X, Y)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('Dens optflow.avi',fourcc, 20.0, (X, Y))

alpha2 = 1/4.5
#gamma_table = np.array([((i/255.0) ** alpha) * 255.0 if i > 20 else 0 for i in range(0,256)]).astype(np.uint8)
gamma_table2 = np.array([((i/255.0) ** alpha2) * 255.0 if i > 1 else 0 for i in range(0,256)]).astype(np.uint8)

while(1):
    ret, frame1 = cap.read()

    
    frame2 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    BW = np.array([255 if i > 106 else 0 for i in range(0,256)]).astype(np.uint8)

    kernel = np.ones((3, 3), "uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    frame2 = clahe.apply(frame2)
    
    bilateral_filtered_image = cv2.bilateralFilter(frame2, 5, 175, 175)
    cv2.imshow('Bilateral', bilateral_filtered_image)
    
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 60, 200)
    #edge_detected_image = cv2.dilate(edge_detected_image, kernel,iterations=1)
    #edge_detected_image = cv2.erode(edge_detected_image, kernel,iterations=1)

    cv2.imshow('Edge', edge_detected_image)

    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.005*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 15) & (len(approx) < 35) & (area > 100) ):
            contour_list.append(contour)

    frame2_RGB = cv2.cvtColor(frame2,cv2.COLOR_GRAY2BGR)

    cv2.drawContours(frame2_RGB, contour_list,  -1, (0,255,0), 2)
    
    cv2.imshow('Objects Detected',frame2_RGB)
 

    frame3 = cv2.LUT(frame2,BW)
    frame4 = edge_detected_image #cv2.LUT(frame3,gamma_table2)
   
    #cv2.imshow('frame4',frame2)

    next = edge_detected_image
    
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

#*********************************************************{ optical flow of Horizontal and Vertical vectors}************************
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')


    #cv2.imshow('Horizontal Component', horz)
    #cv2.imshow('Vertical Component', vert)
#***********************************************************************************************************************************

    #cv2.imshow('RGB',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next
    out.write(rgb)
cap.release()
cv2.destroyAllWindows()
