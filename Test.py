import keras
from keras import backend as K
import json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import os
from Code.network.segnet.custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
from Code.network.unetmod.u_net_mod import BilinearUpSampling2D
import tensorflow as tf
import cv2
import numpy as np
from skimage.util.shape import view_as_windows
import math



def createTiles(img):
    size = img.shape[0]
    sizeNew = 0
    for x in range(0,100):
        if size<2**x:
            sizeNew = 2**x
            break

    pad = sizeNew - size
    imgNew = np.zeros((sizeNew,sizeNew,3))
    imgNew[pad//2:sizeNew-(pad//2),pad//2:sizeNew-(pad//2),:] = img

    #save_img("Check.png", imgNew)

    new_imgs = view_as_windows(imgNew, (patch_width, patch_height, 3), (patch_width, patch_height, 3))
    new_imgs = new_imgs.reshape(-1,patch_width, patch_height, 3)
    #print(new_imgs.shape)

    return new_imgs

def mergeTiles(tiles):
    #print(int(math.sqrt(tiles.shape[0])))
    num = int(math.sqrt(tiles.shape[0]))
    img = np.zeros((patch_width*num,patch_height*num,1))
    count = 0
    for x in range(0,num):
        for y in range(0,num):
            startX = (x)*(patch_width)
            endX = (x+1)*(patch_width)
            startY = (y)*(patch_width)
            endY = (y+1)*(patch_width)

            img[startX:endX,startY:endY,:]= tiles[count]
            count = count + 1
    
    pad = img.shape[0]-im_width
    img = img[pad//2:im_width+(pad//2),pad//2:im_width+(pad//2),:]
    return img


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with open('./config.json') as config_file:
    config = json.load(config_file)


im_width = config['im_width']
im_height = config['im_height']
patch_width = config['patch_width']
patch_height = config['patch_height']

if config['Model'] == "UNETMOD":
    model = keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False,
              custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D(target_shape=(256,16,16),data_format=K.image_data_format())})

if config['Model']=="UNET":
    model = keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False)
if config['Model']=="SEGNET":
    model = keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False,
              custom_objects={'MaxPoolingWithIndices':MaxPoolingWithIndices,'UpSamplingWithIndices':UpSamplingWithIndices})
if config['Model']=="DEEPLAB":
    model = tf.keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False, custom_objects={'tf': tf})

#print(model.summary())
#config['sample_test_image']
#config['sample_test_mask']
image = img_to_array(load_img(config['sample_test_image'], color_mode='rgb', target_size=[config['im_width'],config['im_height']]))/255.0
mask = img_to_array(load_img(config['sample_test_mask'], color_mode='grayscale', target_size=[config['im_width'],config['im_height']]))/255.0

image = createTiles(image)


#print(image.shape)
pred = model.predict(image)
#print(pred.shape)

pred = mergeTiles(pred)
#print(pred.shape)
img_array = pred
#img_array = img_to_array(pred)
# save the image with a new filename

base=os.path.basename(config['sample_test_image'])
fn = os.path.splitext(base)[0]
filename = './Results/outputs/'+fn+'.jpg'
save_img(filename, img_array*255.0)
print("The Output mask is stored at "+ filename)

