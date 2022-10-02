import cv2
import numpy as np
import urls

webcam = False
path = 'camera.jpg'
cam = cv2.VideoCapture(0)

cam.set(10,160)
cam.set(3,1920)
cam.set(4,1080)
scale = 3
wp = 210 *scale
hp = 297 *scale



while True:
    if webcam: _, frame = cam.read()
    else: frame = cv2.imread(path)
    
    
    frame, conts =  urls.getContour(frame, minArea = 50000,filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = urls.warpImg(frame, biggest,wp,hp)
        cv2.imshow('treo',imgWarp)

    frame = cv2.resize(frame,(0,0),None,0.5,0.5)
    cv2.imshow('RRR',frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
