import glob
import numpy as np
import matplotlib.pyplot as plt


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, nadam,SGD
from keras.layers import Input
from UNET_Model.U_Net import get_unet
from Utils.LossFunctions import jaccard_distance_loss
from Utils.MetricFunctions import iou,dice_coef

# Set some parameters
im_width = 1024
im_height = 1024
Epochs = 1


TRAIN_PATH_IMAGES = 'MonuSeg/Training/TissueImages/*'
TRAIN_PATH_GT = 'MonuSeg/Training/GroundTruth/*'
TEST_PATH_IMAGES = 'MonuSeg/Test/TissueImages/*'
TEST_PATH_GT = 'MonuSeg/Test/GroundTruth/*'


ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
ids_train_y = glob.glob(TRAIN_PATH_GT)
print("No. of training images = ", len(ids_train_x))
ids_test_x = glob.glob(TEST_PATH_IMAGES)
ids_test_y = glob.glob(TEST_PATH_GT)
print("No. of testing images = ", len(ids_test_x))

X_train = np.zeros((len(ids_train_x), im_height, im_width, 3), dtype=np.float32)
y_train = np.zeros((len(ids_train_y), im_height, im_width, 1), dtype=np.float32)

X_test = np.zeros((len(ids_test_x), im_height, im_width, 3), dtype=np.float32)
y_test = np.zeros((len(ids_test_y), im_height, im_width, 1), dtype=np.float32)


print("Loading Training Data")
count =0 
for x in (ids_train_x):
    y = glob.glob(x[:-4]+'*')[0]
    x_img = img_to_array(load_img(x, color_mode='rgb', target_size=[im_width,im_height]))
    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale', target_size=[im_width,im_height]))
    X_train[count] = x_img/255.0
    y_train[count] = mask/255.0
    count = count+1


print("Loading Testing Data")
count =0 
for x in (ids_test_x):
    y = glob.glob(x[:-4]+'*')[0]
    x_img = img_to_array(load_img(x, color_mode='rgb', target_size=[im_width,im_height]))
    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale', target_size=[im_width,im_height]))
    X_test[count] = x_img/255.0
    y_test[count] = mask/255.0
    count = count+1
  
 
input_img = Input((im_width, im_height, 3), name='img')
model = get_unet(input_img, n_filters=32, dropout=0.05, batchnorm=True)
#model.compile(optimizer=sgd(), loss="binary_crossentropy"dice_coef_loss,jaccard_distance_loss metrics=["accuracy"]) # ,f1_m,iou_coef,dice_coef
model.compile(optimizer=SGD(lr=1e-5, momentum=0.95), loss=jaccard_distance_loss, metrics=[iou,dice_coef])
print (model.summary())
#nadam(lr=1e-5)
#Adam(1e-5, amsgrad=True, clipnorm=5.0)
#Adam()
#SGD(lr=1e-5, momentum=0.95)
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('./Weights/UNETU-Net-Best.h5', verbose=1, save_best_only=True, save_weights_only=False)
]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


results = model.fit(X_train, y_train, batch_size=1, verbose=1, epochs=Epochs, callbacks=callbacks,\
                    validation_data=(X_test, y_test))

          
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig('./Plots/UNET/train_loss.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["dice_coef"], label="dice_coef")
plt.plot(results.history["val_dice_coef"], label="val_dice_coef")
plt.plot( np.argmax(results.history["val_dice_coef"]), np.max(results.history["val_dice_coef"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Dice Coeff")
plt.legend();
plt.savefig('./Plots/UNET/train_dice.png')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["iou"], label="iou")
plt.plot(results.history["val_iou"], label="val_iou")
plt.plot( np.argmax(results.history["val_iou"]), np.max(results.history["val_iou"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("iou")
plt.legend();
plt.savefig('./Plots/UNET/train_iou.png')

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
