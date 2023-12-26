'''
Data-Augmenter: It adjusts brightness, contrast, rotation, widht- and height-shift, and zoom. It also adds noise and changes some images to grayscale
'''
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def augment(image):
    if tf.random.uniform((), minval=0, maxval=1)<0.1:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1,1,3])
    image = tf.image.random_brightness(image, max_delta=0.25)
    image = tf.image.random_contrast(image, lower=0.85, upper=1)
    image = tf.image.random_flip_left_right(image)
    noise = np.random.normal(loc=0, scale=0.018, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def create_data_augmenter(train_images):
    # Create an instance of the ImageDataGenerator class for data augmentation
    data_augmenter = ImageDataGenerator(
        rotation_range=9,  # rotate the image up to 8 degrees
        width_shift_range=0.05,  
        height_shift_range=0.05,  
        zoom_range=0.1,  # zoom in/out up to 10%
        preprocessing_function=augment  # Add the augment function as the preprocessing function
)
    return data_augmenter