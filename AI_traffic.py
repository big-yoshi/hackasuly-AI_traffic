#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import layers
from keras import backend as K
from keras.preprocessing.image import *
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.applications.vgg19 import *
import os
try:
    from imageai.Detection import ObjectDetection
except:
    get_ipython().system('pip3 install imageai')


# # defining directories

# In[2]:


from PIL import Image
import random
import cv2
demo_dataset = "AI_traffic_dataset\\"
model_name = 'resnet50.h5'


# # Loading Models

# In[3]:


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_name)
detector.loadModel()
vgg19 = VGG19()


def predict_frame(frame):
    all_preds = {}
    K.set_learning_phase(0)
    # for VGG
    vgg_frame = Image.open(frame)
    # for RESNET
    resent_frame = detector.detectObjectsFromImage(input_image=frame)
    # preprocessing the frame
    img_w_boxes = img_to_array(vgg_frame)
    
    for objs in resent_frame:
        name = objs['name']
        box_points = objs['box_points']
        prob_percentage  = objs['percentage_probability']
        
        if name == 'person':
            cropped_img = Image.open(frame)
            cropped_image = cropped_img.crop(box_points)
            cropped_image = img_to_array(cropped_image)
            cropped_image = cv2.resize(cropped_image,(224,224))
            cropped_image = cropped_image.reshape((1,224,224,3))
            img_ = preprocess_input(cropped_image)
            pred = vgg19.predict(img_)
            vgg_pred = decode_predictions(pred,top=3)[0]
            vgg_pred = [(label,prediction) for n,label,prediction in vgg_pred]
        else:
            vgg_pred = ''
        
        try:
            all_preds[name]
        except:
            all_preds[name] = 0
        
        all_preds[name] +=1
        # (top,left), (right,bottom)
        valid_pts = (box_points[0],box_points[1]),(box_points[2],box_points[3])
        # randomizing RGB colors with thickness of 2 for the rectangle
        cv2.rectangle(img_w_boxes,valid_pts[0],valid_pts[1],(random.randrange(0,255),random.randrange(0,255),random.randrange(0,255)),2)
        cv2.putText(img_w_boxes,str(name),(box_points[0],box_points[1]),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),1,cv2.LINE_AA)
        prob_str = "%.1f"%prob_percentage

        if name=="person":
            print("Detected:",name,":",prob_str,":",vgg_pred)
        else:
            print("Detected:",name,":",prob_str)
        
    
    converting_back_2_img = cv2.cvtColor(img_w_boxes,cv2.COLOR_RGB2BGR)
    cv2.imwrite('test.jpg',converting_back_2_img)
    screen = load_img("test.jpg")
    plt.imshow(screen)
    plt.plot()


# In[55]:


"""imgs = os.listdir(demo_dataset)[2]
print(os.listdir(demo_dataset))
img_dir = demo_dataset+imgs
print(img_dir)
predict_frame(img_dir)"""

def _frame(frame,vid_name):
    
    K.set_learning_phase(0)
    # for VGG
    f_frame = Image.open(frame)
    # for RESNET
    resent_frame = detector.detectObjectsFromImage(input_image=frame)
    # preprocessing the frame
    img_w_boxes = img_to_array(f_frame)
    
    for objs in resent_frame:
        name = objs['name']
        box_points = objs['box_points']
        prob_percentage  = objs['percentage_probability']
        
        
        
        # (top,left), (right,bottom)
        valid_pts = (box_points[0],box_points[1]),(box_points[2],box_points[3])
        # randomizing RGB colors with thickness of 2 for the rectangle
        cv2.rectangle(img_w_boxes,valid_pts[0],valid_pts[1],(random.randrange(0,255),random.randrange(0,255),random.randrange(0,255)),2)
        cv2.putText(img_w_boxes,str(name),(box_points[0],box_points[1]),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),1,cv2.LINE_AA)
        #prob_str = "%.1f"%prob_percentage
    
    converting_back_2_img = cv2.cvtColor(img_w_boxes,cv2.COLOR_RGB2BGR)
    full_path = 'vid_frames\\'+str(vid_name)+'.jpg'
    cv2.imwrite(full_path,converting_back_2_img)