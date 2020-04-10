import os
import random

import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.layers import merge


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

def get_unet_mod(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
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
    """
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
    """
    c9 = aspp_block(c5,num_filters=256,rate_scale=1,output_stride=16,input_shape=(256,256,3))
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def aspp_block(x,num_filters=256,rate_scale=1,output_stride=16,input_shape=(256,256,3)):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv3_3_1 = ZeroPadding2D(padding=(6*rate_scale, 6*rate_scale))(x)
    conv3_3_1 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(6*rate_scale, 6*rate_scale),padding='valid')(conv3_3_1)
    conv3_3_1 = BatchNormalization(axis=bn_axis)(conv3_3_1)
    
    conv3_3_2 = ZeroPadding2D(padding=(12*rate_scale, 12*rate_scale))(x)
    conv3_3_2 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(12*rate_scale, 12*rate_scale),padding='valid')(conv3_3_2)
    conv3_3_2 = BatchNormalization(axis=bn_axis)(conv3_3_2)
    
    conv3_3_3 = ZeroPadding2D(padding=(18*rate_scale, 18*rate_scale))(x)
    conv3_3_3 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(18*rate_scale, 18*rate_scale),padding='valid')(conv3_3_3)
    conv3_3_3 = BatchNormalization(axis=bn_axis)(conv3_3_3)
    

    # conv3_3_4 = ZeroPadding2D(padding=(24*rate_scale, 24*rate_scale))(x)
    # conv3_3_4 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(24*rate_scale, 24*rate_scale),padding='valid')(conv3_3_4)
    # conv3_3_4 = BatchNormalization()(conv3_3_4)
    
    conv1_1 = _conv(filters=num_filters, kernel_size=(1, 1),padding='same')(x)
    conv1_1 = BatchNormalization(axis=bn_axis)(conv1_1)

    global_feat = AveragePooling2D((input_shape[0]/output_stride,input_shape[1]/output_stride))(x)
    global_feat = _conv(filters=num_filters, kernel_size=(1, 1),padding='same')(global_feat)
    global_feat = BatchNormalization()(global_feat)
    global_feat = BilinearUpSampling2D((256,input_shape[0]/output_stride,input_shape[1]/output_stride),factor=input_shape[1]/output_stride)(global_feat)

    # y = merge([conv3_3_1,conv3_3_2,conv3_3_3,conv1_1,global_feat,], mode='concat', concat_axis=3)
    y = concatenate([conv3_3_1,conv3_3_2,conv3_3_3,conv1_1,global_feat])
    """
    y = merge([
        conv3_3_1,
        conv3_3_2,
        conv3_3_3,
        # conv3_3_4,
        conv1_1,
        global_feat,
        ], mode='concat', concat_axis=3)
    """
    
    # y = _conv_bn_relu(filters=1, kernel_size=(1, 1),padding='same')(y)
    y = _conv(filters=256, kernel_size=(1, 1),padding='same', name='ASPPLAST')(y)
    y = BatchNormalization()(y)
    return y
    
def _conv(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate',(1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,activation='linear')(input)
        return conv
    return f

class BilinearUpSampling2D(Layer):
    """Upsampling2D with bilinear interpolation."""

    def __init__(self, target_shape=None,factor=None, data_format=None, **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {
            'channels_last', 'channels_first'}
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        self.factor = factor
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0],
                    self.target_size[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1],
                    self.target_size[0], self.target_size[1])

    def call(self, inputs):
        return K.resize_images(inputs, self.factor, self.factor, self.data_format)

    def get_config(self):
        config = {'target_shape': self.target_shape,
                'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))