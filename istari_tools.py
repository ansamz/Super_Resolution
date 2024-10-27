# This script contains helper functions

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def create_dataset(lr_dir, hr_dir, batch_size):
    '''
    create a dataset
    '''
    lr_paths = sorted(os.listdir(lr_dir))
    hr_paths = sorted(os.listdir(hr_dir))
    
    dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
    
    def load_images(lr_path, hr_path):
        lr_img, hr_img = load_and_preprocess(os.path.join(lr_dir, lr_path), os.path.join(hr_dir, hr_path))
        return lr_img, hr_img

    dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
