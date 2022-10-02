import numpy as np
import cv2

def getContour(frame,cThr=[100,100],showCanny = False,minArea=1000,filter=0,draw=False):
    imgGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((5,5))
    imgDail = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv2.erode(imgDail,kernel,iterations=2)
    
    if showCanny:cv2.imshow('canny',imgThre)
    
    contours, heiarchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalcontours = []
    
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            perimetrs = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*perimetrs,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0 :
                if len(approx) == filter:
                    finalcontours.append([len(approx),area,approx,bbox,i])
            else:
                finalcontours.append([len(approx),area,approx,bbox,i])
                
    finalcontours = sorted(finalcontours,key = lambda x:x[1], reverse=True)
    
    if draw:
        for con in finalcontours:
            cv2.drawContours(frame,con[4],-1,(0,0,255),3)
            
    return frame, finalcontours

def reorder(mypoints):
    print(mypoints.shape)
    mypointsnew = np.zeros_like(mypoints)
    mypoints = mypoints.reshape((4,2))
    add = mypoints.sum(1)
    mypointsnew[0] = mypoints[np.argmin(add)]
    mypointsnew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints,axis=1)
    mypointsnew[1] = mypoints[np.argmin(diff)]
    mypointsnew[2] = mypoints[np.argmax(diff)]
    return mypointsnew

def warpImg(frame, points,w,h):
    points = reorder(points)
    
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(frame,matrix,(w,h))
    
    return imgWarp