import keras
import json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import os
from Code.network.segnet.custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with open('./config.json') as config_file:
    config = json.load(config_file)

if config['Model'] == "UNETMOD":
    model = keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False)

if config['Model']=="UNET":
    model = keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False)
if config['Model']=="SEGNET":
    model = keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False,
              custom_objects={'MaxPoolingWithIndices':MaxPoolingWithIndices})
if config['Model']=="DEEPLAB":
    model = tf.keras.models.load_model("./Results/weights/" + config['Model'] + "/" +config['Model']+"-Best.h5", compile=False)

#print(model.summary())
#config['sample_test_image']
#config['sample_test_mask']
image = img_to_array(load_img(config['sample_test_image'], color_mode='rgb', target_size=[config['im_width'],config['im_height']]))/255.0
mask = img_to_array(load_img(config['sample_test_mask'], color_mode='grayscale', target_size=[config['im_width'],config['im_height']]))/255.0
image = image.reshape(1,1024,1024,3)
pred = model.predict(image)
pred = pred.reshape(1024,1024,1)
img_array = img_to_array(pred)
# save the image with a new filename

base=os.path.basename(config['sample_test_image'])
fn = os.path.splitext(base)[0]
filename = './Results/outputs/'+fn+'.jpg'
save_img(filename, img_array*255.0)
print("The Output mask is stored at "+ filename)
