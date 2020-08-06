from keras.models import Model
from keras.layers import Activation,Input,ZeroPadding2D,Cropping2D
from .custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
#import config as cf

def get_segnet(image_shape):
    padding=((0,0),(0,0))
    #image_shape=(1024,1024,3)
    num_classes = 1
    inputs=Input(shape=image_shape)

    x = ZeroPadding2D(padding)(inputs)

    x=CompositeConv(x,2,64)
    x,argmax1=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,2,64)
    x,argmax2=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,3,64)
    x,argmax3=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=CompositeConv(x,3,64)
    x,argmax4=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=CompositeConv(x,3,64)
    x,argmax5=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=UpSamplingWithIndices()([x,argmax5])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax4])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax3])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax2])
    x=CompositeConv(x,2,64)
    
    x=UpSamplingWithIndices()([x,argmax1])
    x=CompositeConv(x,2,[64,num_classes])

    x=Activation('sigmoid')(x)

    y=Cropping2D(padding)(x)
    my_model=Model(inputs=inputs,outputs=y)
    
    return my_model

