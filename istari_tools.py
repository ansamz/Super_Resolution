# This script contains helper functions

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
from tensorflow.image import ssim, psnr
import numpy as np
import random


def load_image(image_path, img_size):

    img = Image.open(image_path)
    img = img.resize(img_size)
    
    # convert to numpy array
    img = np.array(img)
    
    # normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    
    # add channel dimension if needed
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    
    return img


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

    # Resize HR after prediction, not during preprocessing
    hr_img = Image.open(hr_path)
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
    
    # Resize LR if needed (depends on your image sizes)
    lr_img = tf.image.resize(lr_img, [image_size // scale_factor, image_size // scale_factor]) 
    
    return lr_img, hr_img


def create_dataset(lr_dir, hr_dir, batch_size, image_size, scale_factor, num_train_images):
    '''
    create a dataset
    '''
    # Ensure files exist 
    if not os.path.exists(lr_dir) or not os.path.exists(hr_dir):
        raise FileNotFoundError("One or more directories not found.")

    lr_images = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])[:num_train_images]
    hr_images = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])[:num_train_images]

    train_dataset = tf.data.Dataset.from_tensor_slices((lr_images, hr_images))
    train_dataset = train_dataset.map(
        lambda x, y: load_and_preprocess_valid_data(x, y, image_size, scale_factor),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset


# Create a custom dataset with data generators (for efficient memory usage)
def data_generator(lr_dir, hr_dir, batch_size, image_size, scale_factor):
    lr_files = [os.path.join(lr_dir, f) for f in os.listdir(lr_dir)]
    hr_files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir)]
    
    while True:  # Infinite loop for continuous training
        batch_lr = []
        batch_hr = []
        for _ in range(batch_size):
            random_index = random.randint(0, len(lr_files) - 1)
            lr_path, hr_path = lr_files[random_index], hr_files[random_index]
            
            try:  # Add error handling for file reads
                lr_img, hr_img = load_and_preprocess(lr_path, hr_path, image_size, scale_factor)
                batch_lr.append(lr_img)
                batch_hr.append(hr_img)
            except Exception as e:
                print(f"Error loading images: {e}")
        yield np.array(batch_lr), np.array(batch_hr)


def create_test_dataset(test_lr_paths, test_hr_paths, test_hr_dir, test_lr_dir, image_size, scale_factor):
    test_dataset = tf.data.Dataset.from_tensor_slices((test_lr_paths, test_hr_paths))

    def load_images(lr_path, hr_path):
        lr_img, hr_img = load_and_preprocess(os.path.join(test_lr_dir, lr_path), os.path.join(test_hr_dir, hr_path), image_size, scale_factor)
        return lr_img, hr_img

    test_dataset = test_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(1)
    return test_dataset


def calculate_metrics(hr_images, sr_images, metrics=['PSNR', 'SSIM']):
    """Calculate various image quality metrics"""
    results = {}
    for metric in metrics:
        if metric == 'PSNR':
            values = []
            for hr, sr in zip(hr_images, sr_images):
                # Convert to float32 and ensure range [0, 1]
                hr = tf.cast(hr, tf.float32) / 255.0
                sr = tf.cast(sr, tf.float32) / 255.0
                values.append(float(psnr(hr, sr, max_val=1.0)))
            results['PSNR'] = np.mean(values)
        elif metric == 'SSIM':
            values = []
            for hr, sr in zip(hr_images, sr_images):
                # Convert to float32 and ensure range [0, 1]
                hr = tf.cast(hr, tf.float32) / 255.0
                sr = tf.cast(sr, tf.float32) / 255.0
                values.append(float(ssim(hr, sr, max_val=1.0)))
            results['SSIM'] = np.mean(values)
        else:
            print(f"Unsupported metric: {metric}")
    
    return results
