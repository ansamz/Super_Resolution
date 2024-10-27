# this script will contain modeling functions

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def residual_block(inputs, filters, kernel_size, use_bias=False):
    '''
    create the residual block used in RDN, RRDN, and EDSR
    '''
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([inputs, x])


def build_rdn(image_size, num_blocks=8, growth_rate=32, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))
    
    # Feature extraction
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

    # Dense blocks
    for i in range(num_blocks):
        x = residual_block(x, growth_rate, 3)

    # Global context feature fusion
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)

    # Reconstruction layers
    for i in range(int(np.log2(scale_factor)) - 1):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # Final upsampling and convolution
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='sigmoid')(x)

    return Model(input_tensor, x)


def build_rrdn(image_size, num_blocks=16, growth_rate=64, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))
    
    # Shallow feature extraction
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

    # Dense blocks
    for i in range(num_blocks):
        x = residual_block(x, growth_rate, 3)

    # UpSampling with residual learning
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Final convolution and upsampling
    x = layers.Conv2D(num_channels, 3, padding='same', activation='sigmoid')(x)
    x = layers.UpSampling2D()(x)

    return Model(input_tensor, x)


def build_edsr(image_size, num_blocks=16, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))
    
    # Feature extraction
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

    # Residual blocks
    for i in range(num_blocks):
        x = residual_block(x, 64, 3)

    # Final convolutions and upsampling
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    for i in range(int(np.log2(scale_factor))):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='sigmoid')(x)

    return Model(input_tensor, x)


def build_srgan_generator(image_size, num_blocks=16, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))

    # Feature extraction
    x = layers.Conv2D(64, 9, padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', strides=2)(x)

    # Residual blocks
    for i in range(num_blocks):
        x = residual_block(x, 128, 3)

    # Upsampling layers
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # Final convolution
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_real_esrgan_generator(image_size, num_blocks=23, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))

    # Initial convolution
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

    # Residual blocks
    for i in range(num_blocks):
        x = residual_block(x, 64, 3)

    # Upsampling layers
    for i in range(int(np.log2(scale_factor))):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # Final convolution
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_cyclegan_generator(image_size, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))

    # DownSampling layers
    x = layers.Conv2D(64, 7, padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', strides=2)(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', strides=2)(x)
    
    # Reshape for dense blocks
    x = layers.Reshape((image_size // (scale_factor * 2), image_size // (scale_factor * 2), 128 * 4))(x)
    
    # Dense blocks
    for _ in range(9):
        x = residual_block(x, 128 * 4, 3)

    # Upsampling layers
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    
    # Final convolution
    x = layers.Conv2D(num_channels, 7, padding='same', activation='tanh')(x)
    
    return Model(input_tensor, x)


def build_drct_decoder(image_size, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, 128))
    
    # Upsampling layers
    x = layers.UpSampling2D()(input_tensor)
    x = layers.Conv2D(128, 5, padding='same', activation='relu')(x)
    
    for _ in range(3):
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.UpSampling2D()(x)

    # Final convolution
    x = layers.Conv2D(num_channels, 5, padding='same', activation='sigmoid')(x)

    return Model(input_tensor, x)
