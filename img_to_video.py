import cv2
import numpy as np
import glob
 
fps = 10
vid_lenght = fps-2



# list for image data
img_array = []
for filename in glob.glob('vid_frames/*.jpg'):
    img = cv2.imread(filename)
    im_shape = img.shape
    size = (im_shape[1],im_shape[0])
    img_array.append(img)
print('video will be',vid_lenght,'seconds long fps:',fps,"amount of frames in the dir:",len(img_array)) 
out = cv2.VideoWriter('full-vid/full_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(img_array)):
    cv2.waitKey(100)
    out.write(img_array[i])
out.release()