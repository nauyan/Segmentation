from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import cv2
import sys
import numpy
import os
numpy.set_printoptions(threshold=sys.maxsize)
ids_train_x = glob.glob('./Datasets/MonuSeg/Training/TissueImages/*')
ids_train_y = glob.glob('./Datasets/MonuSeg/Training/GroundTruth/*')
#print(ids_train_x)
for x in (ids_train_x):
    base=os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob('./Datasets/MonuSeg/Training/GroundTruth/'+fn+'*')[0]
    print(y)
    #print(x[:-4])
    
    #x_img = img_to_array(load_img(x, color_mode='rgb', target_size=[1024,1024]))
    mask = img_to_array(load_img(y, color_mode='grayscale', target_size=[1024,1024]))
    #mask = img_to_array(load_img(y, color_mode='rgb'))
    

#mask = cv2.imread(y) 
#print(mask)
print(numpy.unique(mask, return_counts=True))









