from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from keras.models import load_model
from keras.optimizers import Adam, nadam,sgd
from skimage.transform import resize
import keras
import numpy as np
from PIL import Image 
from PIL import Image, ImageOps

#GroundTruth#TissueImages
#mask = Image.open("../MonuSeg/Test/GroundTruth/TCGA-ZF-A9R5-01A-01-TS1_bin_mask.png")#.convert('L')
#mask = img_to_array(mask)
#temp = img_to_array(load_img("../MonuSeg/Test/GroundTruth/TCGA-ZF-A9R5-01A-01-TS1_bin_mask.png", color_mode='grayscale', target_size=[1024,1024]))
#load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             #interpolation='nearest')grayscale
#print(temp.shape)
#print(np.unique(mask, return_counts=True))
#mask = ImageOps.invert(mask)
#mask = np.array(mask)
#mask[mask <= 40] = 1
#mask[mask > 40] = 0

#print(np.unique(mask, return_counts=True))
#print(mask.shape)
#print(np.unique(mask[:,:,1], return_counts=True))
#print(np.unique(mask[:,:,2], return_counts=True))
#save_img('test.png',temp)#.reshape(1000,1000,3)
#mask = mask.resize((1024,1024))

#img = img_to_array(load_img("../MonuSeg/Test/TissueImages/TCGA-ZF-A9R5-01A-01-TS1.tif"))
#img = resize(img, (1, 1024, 1024, 3), mode = 'constant')#, preserve_range = True)
#img keras.layers.Reshape(target_shape)

#mask = img_to_array(load_img("../MonuSeg/Test/GroundTruth/TCGA-ZF-A9R5-01A-01-TS1_bin_mask.png", color_mode='grayscale', target_size=[1024,1024]))
#print(np.unique(mask, return_counts=True))
#mask = mask/255.0
#print(np.unique(mask, return_counts=True))
# Opens a image in RGB mode  
im1 = Image.open("../MonuSeg/Test/TissueImages/TCGA-ZF-A9R5-01A-01-TS1.tif")
im1 = img_to_array(load_img("../MonuSeg/Test/TissueImages/TCGA-ZF-A9R5-01A-01-TS1.tif", color_mode='rgb', target_size=[1024,1024]))
im1 = im1/255.0
model = load_model('U-Net-Best.h5', compile = False)
output = model.predict(im1.reshape(1,1024,1024,3))
output = output/255.0
save_img('out.jpg', output.reshape(1024,1024,1))
#im1 = im1.resize((1024,1024))
#model = load_model('U-Net-Best.h5', compile = False)
#im1 = img_to_array(im1)
#im1 = im1/255.0
#output = model.predict(im1.reshape(1,1024,1024,3))
#print(np.unique(output))
#output = output*255.0
#save_img('out0.jpg', output[:,:,:,0].reshape(1024,1024,1).astype(np.uint8))
#save_img('out1.jpg', output[:,:,:,1].reshape(1024,1024,1).astype(np.uint8))
#save_img('out2.jpg', output[:,:,:,2].reshape(1024,1024,1).astype(np.uint8))
#im1 = im1.save("geeks.jpg") 
 
#model = load_model('U-Net-Best.h5', compile = False)
#model.compile(optimizer=nadam(lr=1e-5), loss=jaccard_distance_loss)
#output = model.predict(img)
#output = output.reshape()
#print(output.shape)
#output = output.reshape(1024,1024,3)
#output = resize(output, (1024, 1024, 3), mode = 'constant', preserve_range = True)
#save_img('out.jpg', img.reshape(1024,1024,3))
