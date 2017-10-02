''' This the Object Tracking Code version 1.0.2 , Based on my experinces on previous codes and OpenCV stages, 
Mojtaba, auguest,16,2016 '''

import cv2, math
import numpy as np 
import argparse
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
#List = npzeros()
print (X, Y)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('123.avi',fourcc, 20.0, (X, Y))

frame_number = 0
# check if the file is opened the do the processes
prev_conts=[]
frames_threshold=70
color_rotation=59
lines_history=[]
obj_num1=[]
obj_num2=[]
threshold = 28
max_no_line = 80
alpha = 1/3.0
alpha2 = 1/9.5

gamma_table = np.array([((i/255.0) ** alpha) * 255.0 if i > 20 else 0 for i in range(0,256)]).astype(np.uint8)
gamma_table2 = np.array([((i/255.0) ** alpha2) * 255.0 if i > 7 else 0 for i in range(0,256)]).astype(np.uint8)
object_number = 0


frame_number2 = 0
# check if the file is opened the do tqhe processes
prev_conts2 = []
frames_threshold2 = 40 #7000
color_rotation2 = 59
lines_history2 = []
trackingObjectsList=[]
threshold2 = 25 #25
max_no_line2 = 80
object_number2= 0
running_avg = None
running_avg2 = None
prev_bright = dict()
bright_path = dict()
#colors=[(255,255,randint(0,255)) for x in range(400)]
while vf.isOpened():
   
    frame_number += 1
    frame_number2 += 1

    _, frame = vf.read()
    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range (2000000):
        i+=1
    BW = np.array([255 if i <70 else 0 for i in range(0,256)]).astype(np.uint8)

    kernel = np.ones((3, 3), "uint8")
    #frame = cv2.LUT(frame,BW)
    #frame = cv2.LUT(frame2,gamma_table2)

    #cv2.imshow('f1',frame)
#*********************************************************************************************************
# Tracking Level 1 : Normal Tracking of all objects
#*********************************************************************************************************
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    cl1 = clahe.apply(frame)
    #framehist= cv2.equalizeHist(cl1)
    #cv2.imshow("fhist1",cl1)
    #cv2.imshow("fhist",framehist)
    frame = cv2.LUT(cl1,BW)
    #frame=cv2.erode(frame,kernel)
    #frame=cv2.dilate(frame,kernel,iterations=3)
    frame3=cv2.Laplacian(frame, cv2.CV_8U,ksize=7)
    
    cv2.imshow("frameorigq",frame)
    cv2.imshow("frame3",frame3)
    frame = frame3
    (_,contours2,hierarchy2) = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    new_conts2=[]
    new_bright=dict()
    trackingObjects= dict()
    lines_history2.append([])
    object_count=0
    object_count2=0
    #for prev_center_index, prev_center in enumerate(prev_conts):
    #    if frame_number - prev_center[3] > frames_threshold:
    prev_conts2 = [prev_center2 for prev_center2 in prev_conts2 if frame_number2-prev_center2[3] <= frames_threshold2]
    for pic2, contour2 in enumerate(contours2):
        area = cv2.contourArea(contour2)
        k = cv2.isContourConvex(contour2)
        if (area > 200) and (area < 1700):#k):
            XX,YY,WW,HH = cv2.boundingRect(contour2)
           # contcenter = ((x+w)/2, (y+h)/2)
            
            cur_center2 = (XX+WW/2 , YY+HH/2)
            #new_conts.append(cur_center)
            if len(prev_conts2) > 0:
                prev_closest2 = prev_conts2[0]
                prev_closest_index2 = 0
                for prev_center_index2, prev_center2 in enumerate(prev_conts2):
                    if (prev_center2[0]-cur_center2[0])**2 + (prev_center2[1]-cur_center2[1])**2 < \
                        (prev_closest2[0]-cur_center2[0])**2 + (prev_closest2[1]-cur_center2[1])**2:
                        prev_closest2=prev_center2
                        prev_closest_index2 = prev_center_index2
                #draw line between cur_center and prev_closest
                #frame = cv2.line(frame,cur_center,prev_closest,(0, 0, 255), 2)
                
                if math.sqrt((prev_closest2[0]-cur_center2[0])**2 + (prev_closest2[1]-cur_center2[1])**2) <= threshold2 and frame_number2-prev_closest2[3] <= frames_threshold2:
                  #  if len(lines_history) <= max_no_line:
                    cur_center2=(cur_center2[0],cur_center2[1],prev_closest2[2],frame_number2)
                    prev_conts2[prev_closest_index2] = cur_center2
                    lines_history2[-1].append((cur_center2,prev_closest2))
                else:
                    cur_center2=(cur_center2[0],cur_center2[1],object_number2, frame_number2)
                    prev_conts2.append(cur_center2)
                    #new_conts.append(cur_center)
                    if object_number2==31:
                        print(math.sqrt((prev_closest2[0]-cur_center2[0])**2 + (prev_closest2[1]-cur_center2[1])**2),prev_closest2[2],frame_number,prev_closest2[3],  frame_number2-prev_closest2[3] <= frames_threshold2)
                        print("here1")
                        #sys.exit()
                    object_number2+=1

            else:
                    cur_center2=(cur_center2[0],cur_center2[1],object_number2, frame_number2)
                    prev_conts2.append(cur_center2)
                    if object_number2==31:
                        print("here2")
                        #sys.exit()
                    object_number2+=1
            #print(object_number2)
            num_pixels=1
            num_pixels_original=1
            avg=0
            avgOriginal=0
            for x in range(XX,XX+WW):
                for y in range(YY,YY+HH):
                   # if  frame[y,x] > 0:
                   #     avg+= frame[y,x]
                    #    florescent[y,x]=255
                    #    num_pixels+=1
                    if frame[y,x] >0 :
                        avgOriginal += frame[y,x]
                        num_pixels_original+=1

            #if num_pixels > 0: 
            #    avg/=float(num_pixels)
            #if cur_center2[2] ==51:
            #    print(cur_center2[2], avg)
            avg = float(avg/num_pixels)
            avgOriginal = float(avgOriginal/num_pixels_original)
            trackingObjects[cur_center2[2]]=(cur_center2[0],cur_center2[1],avgOriginal)
            
            object_count2+=1
            color2=cv2.cvtColor(np.uint8([[[(cur_center2[2]*color_rotation ) % 180,255,255]]]),cv2.COLOR_HSV2BGR)
            color2=color2[0][0]
            color2=(int(color2[0]),int(color2[1]),int(color2[2]))
            frame = cv2.rectangle(frame, (XX, YY), (XX+WW, YY+HH), color2, 2)    
            cv2.putText(frame, str(cur_center2[2]), (XX, YY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2)
            ##check if it is bright enough on 2nd gamma level
            track = False
            
            if cur_center2[2] in prev_bright:
                track = True
            else:
                if (num_pixels>= 10 and avg>= 10): #brightness_threshold:
                    #track in 2nd level
                    track=True
            if track:
                new_bright[cur_center2[2]] = cur_center2 
                if cur_center2[2] in bright_path:
                    bright_path[cur_center2[2]].append(cur_center2)
                else:
                    bright_path[cur_center2[2]]=[cur_center2]
                object_count+=1
                frame = cv2.rectangle(frame, (XX, YY), (XX+WW, YY+HH), color2, 2)    
                cv2.putText(frame, str(cur_center2[2]), (XX, YY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2)
            
    prev_bright=new_bright
    obj_num1.append(object_count)
    obj_num2.append(object_count2)
    #print(object_count,object_count2)

    for obj in bright_path:
        #if obj in new_bright:
            path = bright_path[obj]
            if len(path) > 1:
                last_cen = path[-1]
                if frame_number - last_cen[3] < frames_threshold:
                    for cen in path[-2::-1]:
                        if frame_number - cen[3] < frames_threshold:
                            color2=cv2.cvtColor(np.uint8([[[(cen[2]*color_rotation ) % 180,255,255]]]),cv2.COLOR_HSV2BGR)
                            color2=color2[0][0]
                            color2=(int(color2[0]),int(color2[1]),int(color2[2]))
                            frame = cv2.line(frame,(last_cen[0],last_cen[1]),(cen[0],cen[1]),color2, 2)      
                            last_cen = cen
    trackingObjectsList.append(trackingObjects)
    """
    if frame_number2 > max_no_line2:
        #discard=lines_added[0]
        lines_history2=lines_history2[1:]
        #lines_added=lines_added[1:]
    
    for lines2 in lines_history2:
        #pass
       for line in lines2:
            color2=cv2.cvtColor(np.uint8([[[(line[1][2]*color_rotation ) % 180,255,255]]]),cv2.COLOR_HSV2BGR)
            color2=color2[0][0]
            color2=(int(color2[0]),int(color2[1]),int(color2[2]))
            frame2 = cv2.line(frame2,(line[0][0],line[0][1]),(line[1][0],line[1][1]),color2, 2)
            #cv2.putText(frame, str(object_number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
    """

    cv2.imshow('frame', frame)
    #cv2.imshow('frame', frame)
    print frame_number
    #cv2.imshow('frame2', frame2)
    
    '''cv2.imshow('frame3', frame3)
    cv2.imshow('trackingf', florescent) 
    cv2.imshow('tracking2', florescent2) #florescent) 
    cv2.imshow('tracking', florescent3)
    cv2.imshow('original', frame)'''
  # out.write(frame)
    #out.write(or)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
out.release()
vf.release()
#print bright_path
print obj_num2
print obj_num1
#ObjList = []
#print(trackingObjectsList)
objectTrackingByNum = dict()
for (frameIndex,frameDict) in enumerate(trackingObjectsList):
    for (objnum,coords) in frameDict.iteritems():
        if objnum not in objectTrackingByNum:
            objectTrackingByNum[objnum] = {frameIndex: coords}
        else:
            objectTrackingByNum[objnum][frameIndex] = coords
#for j in range(100):
#    for i in range(880):
#        if len(objectTrackingByNum[j][i]) == 2 : 
#            ObjList.append((j, i))
print objectTrackingByNum[6]
Drawingdata = {"x":[], "y":[], "Intensity":[], "objectNum":[]}
for label, coord in objectTrackingByNum[6].items():
    Drawingdata["x"].append(coord[0])
    Drawingdata["y"].append(coord[1])
    Drawingdata["Intensity"].append(int(coord[2]))
    Drawingdata["objectNum"].append(label)

# display scatter plot data
#plt.figure(figsize=(10,8))

###plt.scatter(Drawingdata["x"], Drawingdata["y"], marker = 'o')
# add labels
#for label, x, y in zip(Drawingdata["objectNum"], Drawingdata["x"], Drawingdata["y"]):
#    plt.annotate(label, xy = (x, y))
#for label, Intensity in zip(Drawingdata["objectNum"], Drawingdata["Intensity"]):
#    plt.annotate(label, Intensity)
###plt.plot(Drawingdata["objectNum"], Drawingdata["Intensity"])

fig = plt.figure()

ax1 = fig.add_subplot(211)
for label, x, y in zip(Drawingdata["objectNum"], Drawingdata["x"], Drawingdata["y"]):
    ax1.annotate(label, xy = (x, y))
ax1.invert_yaxis()
ax1.scatter(Drawingdata["x"], Drawingdata["y"], marker = 'o')

ax2 = fig.add_subplot(212)
ax2.set_title('Scatter Plot', fontsize=20)
ax2.set_xlabel('frame#', fontsize=15)
ax2.set_ylabel('Intensity', fontsize=15)
ax2.plot(Drawingdata["objectNum"], Drawingdata["Intensity"])

plt.show()
#*********************************************************************************************************
# Tracking Level 2 : Special Tracking of fluorescent objects
#*********************************************************************************************************
"""
# Tracking The Florescent parts
    (_,contours,hierarchy) = cv2.findContours(florescent,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    new_conts=[]
    lines_history.append([])
    object_count=0
    #for prev_center_index, prev_center in enumerate(prev_conts):
    #    if frame_number - prev_center[3] > frames_threshold:
    prev_conts = [prev_center for prev_center in prev_conts if frame_number-prev_center[3] <= frames_threshold]
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 200 :#and area < 12500:
            x,y,w,h = cv2.boundingRect(contour)
           # contcenter = ((x+w)/2, (y+h)/2)
            
            cur_center = (x+w/2 , y+h/2)
            #new_conts.append(cur_center)
            if len(prev_conts) > 0:
                prev_closest=prev_conts[0]
                prev_closest_index = 0
                for prev_center_index, prev_center in enumerate(prev_conts):
                    if (prev_center[0]-cur_center[0])**2 + (prev_center[1]-cur_center[1])**2 < \
                        (prev_closest[0]-cur_center[0])**2 + (prev_closest[1]-cur_center[1])**2:
                        prev_closest=prev_center
                        prev_closest_index = prev_center_index
                #draw line between cur_center and prev_closest
                #frame = cv2.line(frame,cur_center,prev_closest,(0, 0, 255), 2)
                
                if math.sqrt((prev_closest[0]-cur_center[0])**2 + (prev_closest[1]-cur_center[1])**2) <= threshold and frame_number-prev_closest[3] <= frames_threshold:
                  #  if len(lines_history) <= max_no_line:
                    cur_center=(cur_center[0],cur_center[1],prev_closest[2],frame_number)
                    prev_conts[prev_closest_index] = cur_center
                    #cur_center=(cur_center[0],cur_center[1],prev_closest[2])
                    #new_conts.append(cur_center)
                    lines_history[-1].append((cur_center,prev_closest))
                else:
                    cur_center=(cur_center[0],cur_center[1],object_number, frame_number)
                    prev_conts.append(cur_center)
                    #new_conts.append(cur_center)
                    
                    object_number+=1
            else:
                    cur_center=(cur_center[0],cur_center[1],object_number, frame_number)
                    prev_conts.append(cur_center)
                    #new_conts.append(cur_center)
                    
                    object_number+=1
            object_count+=1
            print(object_count, object_count2)
            color=cv2.cvtColor(np.uint8([[[(cur_center[2]*color_rotation ) % 180,255,255]]]),cv2.COLOR_HSV2BGR)
            color=color[0][0]
            color=(int(color[0]),int(color[1]),int(color[2]))
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)    
            cv2.putText(frame, str(cur_center[2]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)

                #print(cur_center,prev_closest)
    #prev_conts=new_conts
    
    if frame_number > max_no_line:
        #discard=lines_added[0]
        lines_history=lines_history[1:]
        #lines_added=lines_added[1:]
    
    for lines in lines_history:
        #pass
        for line in lines:
            color=cv2.cvtColor(np.uint8([[[(line[1][2]*color_rotation ) % 180,255,255]]]),cv2.COLOR_HSV2BGR)
            color=color[0][0]
            color=(int(color[0]),int(color[1]),int(color[2]))
            #color=(0,0,255)
            #print(type(color[2]))
            frame = cv2.line(frame,(line[0][0],line[0][1]),(line[1][0],line[1][1]),color, 2)
            #cv2.putText(frame, str(object_number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)

    #print len(new_conts)
#    List[frame_number] = ((x + w/2), (y + h/2))
#    frame_number = 0
                    
    
#    img = cv2.imread('toxop.jpg', 0)
#    k = 0
#    _, contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
#    print len(contours)
#    centres = []
#    for i in range(len(contours)):
#        moments = cv2.moments(contours[i])
#        M = int(moments['m00'])
#        if M != 0 
#            centres[((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
           
  #print b
    #out.write(frame)

# Showing the videos 
    #kernel = np.ones((3, 3), "uint8")
    #ff=hsv
    #ff = cv2.erode(ff, kernel,iterations=1)
    #ff = cv2.dilate(ff, kernel,iterations=2)  
    #kernel = np.ones((3, 3), "uint8")
    
    cv2.imshow('tracking', frame) #florescent) 
    out.write(frame)
    ##cv2.imshow('tracking', florescent) #
    ##out.write(cv2.cvtColor(florescent, cv2.COLOR_GRAY2BGR))  
  #  cv2.imshow('mask', mask)
  #  cv2.imshow('res', res)
  #   print(List)
""" 
    
