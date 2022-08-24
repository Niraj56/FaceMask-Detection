# =====================================================================
# Date : 21 oct 2021
# Title: model
# Author: Niraj Tiwari
# =====================================================================


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#import matplotlib.pyplot as plt
import numpy as np


data_path = "mobile-net-dataset/"

# =====================================================================
# load image data set
def load_dataset():
        mask_ims = os.listdir( data_path + "s64/0/" )
        no_mask_ims = os.listdir( data_path + "s64/1/" )
        
        n = len(mask_ims) + len(no_mask_ims)
        
        x = np.zeros((n, im_size, im_size, 3))
            
    
        
        y = np.zeros((n))
        
        k = 0
            
            
        for i in range(2905):
            x[k] = load_image( data_path + "s64/0/" + mask_ims[i])#.reshape(( im_size, im_size, 3)) 
            y[k] = 0
            k += 1
            
            x[k] = load_image( data_path + "s64/1/" + no_mask_ims[i])#.reshape(( im_size, im_size, 3))
            y[k] = 1
            k += 1
            
            
        return x, y
        
        
    
#==============================================================================   
# convert image to array
def load_image( path ):
    im_mat = img_to_array( load_img(path, color_mode="rgb") ) / 255
    return im_mat.reshape( (im_size, im_size, 3) )


#==============================================================================
# create, compile and return the model  
def createModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(im_size, im_size, 3)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(im_size, im_size, 3)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    return model
    

if __name__ == '__main__':
    im_size = 64
    im_size_sq = im_size ** 2
    
    x, y = load_dataset()
    train_dataset = tf.data.Dataset.from_tensor_slices(( x, y ))
    train_dataset = train_dataset.shuffle(10000).batch(64)
    print(x.shape, y.shape)

    model = createModel()
    model.fit( train_dataset, epochs=10 )
    model.save( "saved_model/model_mn64" )
    
#    model = tf.keras.models.load_model( "saved_model/model_mn1" )
