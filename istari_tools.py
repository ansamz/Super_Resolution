# This script contains helper functions

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
from tensorflow.image import ssim, psnr
import numpy as np


def display_image_pair(lr_path, hr_path):
    lr_img = Image.open(lr_path)
    hr_img = Image.open(hr_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(lr_img)
    ax1.set_title('Low Resolution')
    ax1.axis('off')
    
    ax2.imshow(hr_img)
    ax2.set_title('High Resolution')
    ax2.axis('off')
    
    plt.show()


def load_and_preprocess(lr_path, hr_path, image_size, scale_factor):
    '''
    load and preprocess images
    '''
    lr_img = Image.open(lr_path)
    lr_img = lr_img.resize((image_size // scale_factor, image_size // scale_factor), Image.BICUBIC)
    lr_img = np.array(lr_img) / 255.0

    hr_img = Image.open(hr_path)
    hr_img = hr_img.resize((image_size, image_size), Image.BICUBIC)
    hr_img = np.array(hr_img) / 255.0

    return lr_img, hr_img


def load_and_preprocess_valid_data(lr_path, hr_path, image_size, scale_factor):
    # Read files using TensorFlow operations
    lr_img = tf.io.read_file(lr_path)
    hr_img = tf.io.read_file(hr_path)
    
    # Decode images
    lr_img = tf.image.decode_png(lr_img, channels=3)
    hr_img = tf.image.decode_png(hr_img, channels=3)
    
    # Convert to float32 and normalize
    lr_img = tf.cast(lr_img, tf.float32) / 255.0
    hr_img = tf.cast(hr_img, tf.float32) / 255.0
    
    # Resize if needed
    lr_img = tf.image.resize(lr_img, [image_size, image_size])
    hr_img = tf.image.resize(hr_img, [image_size * scale_factor, image_size * scale_factor])
    
    return lr_img, hr_img


def create_dataset(lr_dir, hr_dir, batch_size, image_size, scale_factor):
    '''
    create a dataset
    '''
    lr_paths = sorted(os.listdir(lr_dir))
    hr_paths = sorted(os.listdir(hr_dir))
    
    dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
    
    def load_images(lr_path, hr_path):
        lr_img, hr_img = load_and_preprocess(os.path.join(lr_dir, lr_path), os.path.join(hr_dir, hr_path), image_size, scale_factor)
        return lr_img, hr_img

    dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def create_test_dataset(test_lr_paths, test_hr_paths, test_hr_dir, test_lr_dir, image_size, scale_factor):
    test_dataset = tf.data.Dataset.from_tensor_slices((test_lr_paths, test_hr_paths))

    def load_images(lr_path, hr_path):
        lr_img, hr_img = load_and_preprocess(os.path.join(test_lr_dir, lr_path), os.path.join(test_hr_dir, hr_path), image_size, scale_factor)
        return lr_img, hr_img

    test_dataset = test_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(1)
    return test_dataset


def calculate_metrics(hr_images, sr_images):
    """Calculate various image quality metrics"""
    psnr_values = []
    ssim_values = []
    
    for hr, sr in zip(hr_images, sr_images):
        # Convert to float32 and ensure range [0, 1]
        hr = tf.cast(hr, tf.float32) / 255.0
        sr = tf.cast(sr, tf.float32) / 255.0
        
        psnr_val = psnr(hr, sr, max_val=1.0)
        ssim_val = ssim(hr, sr, max_val=1.0)
        
        psnr_values.append(float(psnr_val))
        ssim_values.append(float(ssim_val))
    
    return {
        'PSNR': np.mean(psnr_values),
        'SSIM': np.mean(ssim_values)
    }
