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


def residual_block(inputs, filters, kernel_size, use_bias=True):
    '''
    create the residual block used in RDN, RRDN, and EDSR
    '''
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(inputs.shape[-1], kernel_size, padding='same', use_bias=use_bias)(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([inputs, x])


def build_rdn(image_size, num_blocks=8, growth_rate=64, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))
    
    # feature extraction
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)
    skip = x  # save for global residual connection

    for i in range(num_blocks):
        x = residual_block(x, growth_rate, 3)

    # global context feature fusion
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    x = layers.Add()([x, skip])  # Add global residual connection

    # reconstruction layers
    for i in range(int(np.log2(scale_factor)) - 1):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # upsampling and convolution
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)
    # for the final layer using "tanh" or "sigmoid" instead, as these activations produce outputs in the range [0, 1] or [-1, 1], more suitable for image intensities.

    return Model(input_tensor, x)


def build_rrdn(image_size, num_blocks=16, growth_rate=64, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))
    
    # shallow feature extraction
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

    for i in range(num_blocks):
        x = residual_block(x, growth_rate, 3)

    # upsampling with residual learning
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)
    x = layers.UpSampling2D()(x)

    return Model(input_tensor, x)


def build_edsr(image_size, num_blocks=16, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))
    
    # feature extraction
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

    for i in range(num_blocks):
        x = residual_block(x, 64, 3)

    # convolutions and upsampling
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    for i in range(int(np.log2(scale_factor))):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_srgan_generator(image_size, num_blocks=16, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))

    # feature extraction
    x = layers.Conv2D(64, 9, padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', strides=2)(x)

    for i in range(num_blocks):
        x = residual_block(x, 128, 3)

    # upsampling layers
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_real_esrgan_generator(image_size, num_blocks=23, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

    for i in range(num_blocks):
        x = residual_block(x, 64, 3)

    # upsampling layers
    for i in range(int(np.log2(scale_factor))):
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_cyclegan_generator(image_size, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))

    # downdampling layers
    x = layers.Conv2D(64, 7, padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', strides=2)(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', strides=2)(x)
    
    for _ in range(9):
        x = residual_block(x, 256, 3)

    # upsampling layers
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)

    x = layers.Conv2D(num_channels, 7, padding='same', activation='tanh')(x)
    
    return Model(input_tensor, x)


def build_cyclegan_discriminator(image_size, num_channels=3):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))

    # Discriminator layers
    x = layers.Conv2D(64, 4, strides=2, padding='same', activation='leaky_relu', kernel_initializer='he_normal')(input_tensor)
    x = layers.Conv2D(128, 4, strides=2, padding='same', activation='leaky_relu', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same', activation='leaky_relu', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(512, 4, strides=2, padding='same', activation='leaky_relu', kernel_initializer='he_normal')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    return Model(input_tensor, x)


def build_drct_encoder(image_size, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, num_channels))
    
    # Downsampling layers
    x = layers.Conv2D(64, 5, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, 5, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    
    for _ in range(3):
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

    return Model(input_tensor, x)


def build_drct_decoder(image_size, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size // scale_factor, image_size // scale_factor, 128))
    
    # Upsampling layers
    x = layers.UpSampling2D()(input_tensor)
    x = layers.Conv2D(128, 5, padding='same', activation='relu')(x)
    
    for _ in range(3):
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.UpSampling2D()(x)

    x = layers.Conv2D(num_channels, 5, padding='same', activation='tanh')(x) # Use tanh for output

    return Model(input_tensor, x)
