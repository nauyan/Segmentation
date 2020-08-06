import glob
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, nadam,SGD
from keras.layers import Input
# from Code.utils.lossfunctions import jaccard_distance_loss,dice_coef_loss
from Code.utils.metricfunctions import dice_coef,f1
from Code.utils.lossfunctions import *

#from Code.network.unetmod.u_net_mod import get_unet_mod
from Code.network.unetmod.u_net_mod import *
from Code.network.unet.u_net import get_unet
from Code.network.segnet.segnet import get_segnet
from Code.network.deeplab.deeplab import Deeplabv3
import argparse
import tensorflow as tf

from skimage.util.shape import view_as_windows
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


with open('./config.json') as config_file:
    config = json.load(config_file)
# print (config)
im_width = config['im_width']
im_height = config['im_height']
patch_width = config['patch_width']
patch_height = config['patch_height']
Epochs = config['Epochs']


TRAIN_PATH_IMAGES = config['TRAIN_PATH_IMAGES']
TRAIN_PATH_GT = config['TRAIN_PATH_GT']
TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
TEST_PATH_GT = config['TEST_PATH_GT']


ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
ids_train_y = glob.glob(TRAIN_PATH_GT)
print("No. of training images = ", len(ids_train_x))
ids_test_x = glob.glob(TEST_PATH_IMAGES)
ids_test_y = glob.glob(TEST_PATH_GT)
print("No. of testing images = ", len(ids_test_x))

#X_train = np.zeros((len(ids_train_x), im_height, im_width, 3), dtype=np.float32)
#y_train = np.zeros((len(ids_train_y), im_height, im_width, 1), dtype=np.float32)

#X_test = np.zeros((len(ids_test_x), im_height, im_width, 3), dtype=np.float32)
#y_test = np.zeros((len(ids_test_y), im_height, im_width, 1), dtype=np.float32)

X_train = []
y_train = []
X_test = []
y_test = []

print("Loading Training Data")
count =0 
for x in (ids_train_x):
    base=os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(config['TRAIN_PATH_GT']+fn+'*')[0]
    x_img = img_to_array(load_img(x, color_mode='rgb', target_size=[im_width,im_height]))
    x_img = x_img/255.0
    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale', target_size=[im_width,im_height]))
    mask = mask/255.0
    #X_train[count] = x_img/255.0
    #y_train[count] = mask/255.0
    new_imgs = view_as_windows(x_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))
    #print("Number of Patches")
    #print(new_imgs.shape)
    for patch in new_imgs:
        X_train.append(patch)
    new_masks = view_as_windows(mask, (patch_width, patch_height, 1), (patch_width//2, patch_height//2, 1))
    for patch in new_masks:
        y_train.append(patch)
    count = count+1



print("Loading Testing Data")
count =0 
for x in (ids_test_x):
    base=os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(config['TEST_PATH_GT']+fn+'*')[0]
    x_img = img_to_array(load_img(x, color_mode='rgb', target_size=[im_width,im_height]))
    x_img = x_img/255.0
    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale', target_size=[im_width,im_height]))
    mask = mask/255.0
    #X_test[count] = x_img/255.0
    #y_test[count] = mask/255.0
    new_imgs = view_as_windows(x_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))
    for patch in new_imgs:
        X_test.append(patch)
    new_masks = view_as_windows(mask, (patch_width, patch_height, 1), (patch_width//2, patch_height//2, 1))
    for patch in new_masks:
        y_test.append(patch)
    count = count+1


#print(len(X_train),len(y_train))
#print(len(X_test),len(y_test))
X_train = np.array(X_train) 
y_train = np.array(y_train) 
X_test = np.array(X_test) 
y_test = np.array(y_test) 

input_img = Input((256, 256, 3), name='img')
#from tensorflow.keras.utils.vis_utils import plot_model

if config['Model'] == "UNETMOD":
    print("Loading UNETMOD Model")
    model = get_unet_mod(input_img, n_filters=16, dropout=0.1, batchnorm=True)  #32
    # model.compile(optimizer=Adam(1e-5), loss=jaccard_distance_loss, metrics=[iou,dice_coef])
    model.compile(optimizer=Adam(amsgrad=True), loss=jaccard_distance_loss, metrics=["accuracy", dice_coef, f1])
    print("Printing Model Summary")
    print (model.summary())
    tf.keras.utils.plot_model(model, './Code/network/unetmod/unet_plot.png')

if config['Model'] == "UNET":
    print("Loading UNET Model")
    model = get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True)  
    # model.compile(optimizer=Adam(1e-5), loss=jaccard_distance_loss, metrics=[iou,dice_coef])
    model.compile(optimizer=Adam(amsgrad=True), loss=jaccard_distance_loss, metrics=["accuracy", dice_coef, f1])
    print("Printing Model Summary")
    print (model.summary())
    tf.keras.utils.plot_model(model, './Code/network/unet/unet_plot.png')

if config['Model'] == "SEGNET":
    print("Loading SEGNET Model")
    model = get_segnet((patch_height, patch_width, 3))
        #n_labels=3,
        #kernel=3,
        #pool_size=(2, 2),
        #output_mode="softmax")
    model.compile(optimizer=Adam(amsgrad=True), loss=jaccard_distance_loss, metrics=["accuracy", dice_coef, f1])
    print("Printing Model Summary")
    print (model.summary())
    tf.keras.utils.plot_model(model, './Code/network/segnet/segnet_plot.png')

if config['Model'] == "DEEPLAB":  
    print("Loading DEEPLAB Model")
    model = Deeplabv3(weights=None, input_tensor=None, input_shape=(patch_height, patch_width, 3), classes=1, backbone='xception',
               OS=16, alpha=1., activation='sigmoid')
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss=jaccard_distance_loss, metrics=["accuracy", dice_coef, f1])
    #plot_model(model, to_file='./Code/network/deeplab/deeplab_plot.png', show_shapes=True, show_layer_names=True)
 
print("Compiling Model")
#model.compile(optimizer=sgd(), loss="binary_crossentropy"dice_coef_loss,jaccard_distance_loss metrics=["accuracy"]) # ,f1_m,iou_coef,dice_coef
#


#nadam(lr=1e-5)
#Adam(1e-5, amsgrad=True, clipnorm=5.0)
#Adam()
#SGD(lr=1e-5, momentum=0.95)
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
    ModelCheckpoint('./Results/weights/'+str(config['Model'])+'/'+str(config['Model'])+'-Best.h5', monitor='val_dice_coef',mode = 'max' , verbose=1, save_best_only=True, save_weights_only=False)
]
X_train = X_train.reshape(-1,patch_height,patch_width,3)
y_train = y_train.reshape(-1,patch_height,patch_width,1)
X_test = X_test.reshape(-1,patch_height,patch_width,3)
y_test = y_test.reshape(-1,patch_height,patch_width,1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


results = model.fit(X_train, y_train, batch_size=config['Batch'], verbose=1, epochs=Epochs, callbacks=callbacks,\
                    validation_data=(X_test, y_test))

print(model.evaluate(X_test, y_test, verbose=1))
          
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig('./Results/plots/'+str(config['Model'])+'/train_loss.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["dice_coef"], label="dice_coef")
plt.plot(results.history["val_dice_coef"], label="val_dice_coef")
plt.plot( np.argmax(results.history["val_dice_coef"]), np.max(results.history["val_dice_coef"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Dice Coeff")
plt.legend();
plt.savefig('./Results/plots/'+str(config['Model'])+'/train_dice.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["f1"], label="f1")
plt.plot(results.history["val_f1"], label="val_f1")
plt.plot( np.argmax(results.history["val_f1"]), np.max(results.history["val_f1"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("f1")
plt.legend();
plt.savefig('./Results/plots/'+str(config['Model'])+'/train_f1.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["accuracy"], label="accuracy")
plt.plot(results.history["val_accuracy"], label="val_accuracy")
plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend();
plt.savefig('./Results/plots/'+str(config['Model'])+'/train_accuracy.png')

"""
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
"""
