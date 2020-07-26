
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time
import tqdm
import gc
import os
from keras.utils import to_categorical
dataset_dir = "car_dataset/train/"
labels = os.listdir(dataset_dir)
img_size = tuple((224,224))



class Model():
    def __init__(self):
        self.model = Sequential()
    def load_model(self):
        self.model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu", input_shape=(224,224,3)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(5,5), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=128, kernel_size=(5,5), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(labels), activation='sigmoid'))
        
    

    
    def load_n_preprocess_data(self,data_dir):
        train_imgs = []
        train_labels = []
        for dt in labels:
            full_dir = os.path.join(dataset_dir,dt)
            imgs = os.listdir(full_dir)
            for img in tqdm.tqdm(imgs):
                img = os.path.join(full_dir,img)
                img = load_img(img,target_size=(224,224))
                img = img_to_array(img)
                #img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                img = img/255
                train_imgs.append(img)
                class_num = labels.index(dt)
                train_labels.append(class_num)
        X = np.array(train_imgs)
        y = to_categorical(train_labels)
        del train_imgs
        del train_labels
        gc.collect()
        return X,y
    
    def train(self,X,y,epoch,loss="binary_crossentropy",opt = "adam",t_size=0.2):
        X_train ,x_test, Y_train , y_test = train_test_split(X,y,test_size=t_size,random_state=42)
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        self.model.fit(X_train, Y_train, epochs=epoch, validation_data=(x_test,y_test),batch_size=32)
        
    def predict_image(self,img_path,show_img=False):
        imgs_ = img_path
        
        preprocessed_img = load_img(imgs_,target_size=(224,224))
        im_array = img_to_array(preprocessed_img)
        im_array = np.reshape(im_array,[1,224,224,3])
        im_array /=255
        lbl = labels[(np.argmax(self.model.predict(im_array)))]
        if show_img == True:
            plt.title(label=lbl,loc='center')
            plt.imshow(preprocessed_img)
            plt.plot()
        return lbl
    def save_model(self,model_name):
        if os.path.exists('saved_models') == False:
            os.mkdir('saved_models')
        else:        
            self.model.save('saved_models\\'+model_name)
    def load_trained_model(self,model_name):
        self.model = self.model.load_weights('saved_models\\'+model_name)



if __name__ == '__main__':
    from keras import backend as k
    k.clear_session()
    # loading model class
    _model = Model()
    _model.load_model()
    # if you want to train
    
    #data_dir = 'car_dataset/train/'
    #X,y = _model.load_n_preprocess_data(data_dir)
    #
    #_model.train(X,y,epoch=20,t_size=0.30)
    #_model.save_model('model_epoch20.h5')
    
    #############


    # loading the saved models

    _model.load_trained_model('model_30.h5')

    # predicting
    _model.predict_image('car_dataset/test/crazy_car.jpg',show_img=True)
    