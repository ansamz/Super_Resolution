# reduced blocks and growth rate, smaller scale factor, reduced filters as well
# removed BatchNormalization

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np


def residual_block(inputs, filters, kernel_size, use_bias=True):
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(inputs.shape[-1], kernel_size, padding='same', use_bias=use_bias)(x)
    return layers.Add()([inputs, x])


def build_rdn(image_size, num_blocks=2, growth_rate=8, num_channels=3, scale_factor=4):
    '''
    Growth rate was set up to 64 but due to to memory crashes I reduced it to 16
    as well as replaced the UpSampling2D layer with Conv2DTranspose since it's more memory effiecient
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D
    '''
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))
    
    # feature extraction
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_tensor)
    skip = x  # save for global residual connection

    for i in range(num_blocks):
        x = residual_block(x, growth_rate, 3)

    # global context feature fusion
    x = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
    x = layers.Add()([x, skip])  # Add global residual connection

    # reconstruction layers
    for i in range(int(np.log2(scale_factor)) - 1):
        x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    #    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # upsampling and convolution
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)
    # for the final layer using "tanh" or "sigmoid" instead, as these activations produce outputs in the range [0, 1] or [-1, 1], more suitable for image intensities.

    return Model(input_tensor, x)


def build_rrdn(image_size, num_blocks=2, growth_rate=32, num_channels=3):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_tensor)
    for i in range(num_blocks):
        x = residual_block(x, growth_rate, 3)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    # use Upsampling2D only once
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)
    return Model(input_tensor, x)


def build_edsr(image_size, num_blocks=8, num_channels=3, scale_factor=2):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_tensor)
    
    for i in range(num_blocks):
        x = residual_block(x, 32, 3)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    # simplified Upsampling
    x = layers.Conv2DTranspose(32, 3, strides=scale_factor, padding='same', activation='relu')(x)

    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)
    return Model(input_tensor, x)


def build_srgan_generator(image_size, num_blocks=8, num_channels=3, scale_factor=2):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))

    x = layers.Conv2D(32, 9, padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', strides=2)(x)

    for i in range(num_blocks):
        x = residual_block(x, 32, 3)

    x = layers.Conv2DTranspose(32, 3, strides=scale_factor, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_real_esrgan_generator(image_size, num_blocks=12, num_channels=3, scale_factor=2):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_tensor)

    for i in range(num_blocks):
        x = residual_block(x, 32, 3)

    x = layers.Conv2DTranspose(32, 3, strides=scale_factor, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_channels, 3, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_cyclegan_generator(image_size, num_channels=3, scale_factor=2):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))

    x = layers.Conv2D(32, 7, padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=2)(x)

    for _ in range(4):  # reduced from 9
        x = residual_block(x, 64, 3)

    # Simplified upsampling (single Conv2DTranspose)
    x = layers.Conv2DTranspose(32, 3, strides=scale_factor, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_channels, 7, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)


def build_cyclegan_discriminator(image_size, num_channels=3):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))

    x = layers.Conv2D(32, 4, strides=2, padding='same', activation='leaky_relu')(input_tensor)
    x = layers.Conv2D(64, 4, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return Model(input_tensor, x)


def build_drct_encoder(image_size, num_channels=3):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))
    x = layers.Conv2D(32, 5, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    return Model(input_tensor, x)


def build_drct_decoder(image_size, num_channels=3, scale_factor=4):
    input_tensor = tf.keras.Input(shape=(image_size, image_size, num_channels))
    
    # Upsampling layers
    x = layers.UpSampling2D()(input_tensor)
    x = layers.Conv2D(128, 5, padding='same', activation='relu')(x)
    
    for _ in range(3):
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.UpSampling2D()(x)

    x = layers.Conv2D(num_channels, 5, padding='same', activation='tanh')(x) # Use tanh for output

    return Model(input_tensor, x)
