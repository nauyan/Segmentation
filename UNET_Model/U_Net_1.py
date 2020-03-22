import os
import random
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#%matplotlib inline

#from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, nadam,sgd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def dice_coef_1(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)=  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        y_true = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=2))
        y_pred = K.flatten(y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# Set some parameters
im_width = 1024
im_height = 1024
border = 5



TRAIN_PATH_IMAGES = '../MonuSeg/Training/TissueImages/*'
TRAIN_PATH_GT = '../MonuSeg/Training/GroundTruth/*'
TEST_PATH_IMAGES = '../MonuSeg/Test/TissueImages/*'
TEST_PATH_GT = '../MonuSeg/Test/GroundTruth/*'

import glob
ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
ids_train_y = glob.glob(TRAIN_PATH_GT)
print("No. of training images = ", len(ids_train_x))
ids_test_x = glob.glob(TEST_PATH_IMAGES)
ids_test_y = glob.glob(TEST_PATH_GT)
print("No. of testing images = ", len(ids_test_x))

X_train = np.zeros((len(ids_train_x), im_height, im_width, 3), dtype=np.float32)
y_train = np.zeros((len(ids_train_y), im_height, im_width, 3), dtype=np.float32)

X_test = np.zeros((len(ids_test_x), im_height, im_width, 3), dtype=np.float32)
y_test = np.zeros((len(ids_test_y), im_height, im_width, 3), dtype=np.float32)

print("Loading Training Data")
count =0 
for x in (ids_train_x):
    y = glob.glob(x[:-4]+'*')[0]
    #print(x,y)
    img = load_img(x)
    x_img = img_to_array(img)
    x_img = resize(x_img, (im_width, im_height, 3), mode = 'constant', preserve_range = True)
    # Load masks
    mask = img_to_array(load_img(y))
    #mask = mask[:,:,1]
    mask = resize(mask, (im_width, im_height, 3), mode = 'constant', preserve_range = True)
    # Save images
<<<<<<< HEAD
    #X_train[count] = x_img/255.0 16777216.0 f1_m
    X_train[count] = x_img/255.0
=======
    #X_train[count] = x_img/255.0 16777216.0
<<<<<<< HEAD
    X_train[count] = x_img/255.0
=======
    X_train[count] = x_img/16777216.0
>>>>>>> refs/remotes/origin/master
>>>>>>> 0386883839b5f6d18a2b5557bbace601ba7ff392
    #y_train[count] = mask/255.0
    y_train[count] = mask/255.0
    count = count+1

print("Loading Testing Data")
count =0 
for x in (ids_test_x):
    y = glob.glob(x[:-4]+'*')[0]
    #print(x,y)
    img = load_img(x)
    x_img = img_to_array(img)
    x_img = resize(x_img, (im_width, im_height, 3), mode = 'constant', preserve_range = True)
    # Load masks
    mask = img_to_array(load_img(y))
    #mask = mask[:,:,1]
    mask = resize(mask, (im_width, im_height, 3), mode = 'constant', preserve_range = True)
    # Save images
    #X_test[count] = x_img/255.0 16777216.0
<<<<<<< HEAD
    X_test[count] = x_img/255.0
=======
<<<<<<< HEAD
    X_test[count] = x_img/255.0
=======
    X_test[count] = x_img/16777216.0
>>>>>>> refs/remotes/origin/master
>>>>>>> 0386883839b5f6d18a2b5557bbace601ba7ff392
    #y_test[count] = mask/255.0
    y_test[count] = mask/255.0
    count = count+1
  
 
input_img = Input((im_width, im_height, 3), name='img')
model = get_unet(input_img, n_filters=32, dropout=0.05, batchnorm=True)
#model.compile(optimizer=sgd(), loss="binary_crossentropy"dice_coef_loss, metrics=["accuracy"]) # ,f1_m,iou_coef,dice_coef
model.compile(optimizer=nadam(lr=1e-5), loss=jaccard_distance_loss, metrics=['acc',f1_m,iou_coef,dice_coef])
print (model.summary())
#nadam(lr=1e-5)
#Adam(1e-5, amsgrad=True, clipnorm=5.0)
#Adam()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('U-Net-Best.h5', verbose=1, save_best_only=True, save_weights_only=False)
]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
results = model.fit(X_train, y_train, batch_size=1, verbose=1, epochs=300, callbacks=callbacks,\
                    validation_data=(X_test, y_test))
               
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig('./train_loss.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["dice_coef"], label="dice_coef")
plt.plot(results.history["val_dice_coef"], label="val_dice_coef")
plt.plot( np.argmax(results.history["val_dice_coef"]), np.max(results.history["val_dice_coef"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Dice Coeff")
plt.legend();
plt.savefig('./train_dice.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["iou_coef"], label="iou_coef")
plt.plot(results.history["val_iou_coef"], label="val_iou_coef")
plt.plot( np.argmax(results.history["val_iou_coef"]), np.max(results.history["val_iou_coef"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("iou_coef")
plt.legend();
plt.savefig('./train_iou_coef.png')


plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["acc"], label="acc")
plt.plot(results.history["val_acc"], label="val_acc")
plt.plot( np.argmax(results.history["val_acc"]), np.max(results.history["val_acc"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend();
plt.savefig('./train_acc.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["f1_m"], label="f1_m")
plt.plot(results.history["val_f1_m"], label="val_f1_m")
plt.plot( np.argmax(results.history["val_f1_m"]), np.max(results.history["val_f1_m"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("F1-Score")
plt.legend();
plt.savefig('./train_F1.png')
""""""