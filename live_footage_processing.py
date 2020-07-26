import cv2
import time
from PIL import Image
import numpy as np
from AI_traffic import _frame
import sys

try:
	vid_length = sys.argv[1]
except:
	vid_length = 100 # frames
cap = cv2.VideoCapture('camera.mp4')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imwrite('test_video.jpg',frame)
    frame_to_predict = "test_video.jpg"
    _frame(frame_to_predict,i)
    i+=1
    #print(type(frame))
    #print(frame)
    #frame_to_predict = cv2.imread(frame_to_predict)
    cv2.imshow('test',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if i == vid_length:
    	break
cap.release()
cv2.destroyAllWindows()