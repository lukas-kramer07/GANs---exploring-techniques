import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

def enlarge_and_sharpen(image_path, output_folder, target_size=(300, 300), sharpen_factor=0.7):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Resize the image using OpenCV
    resized_image = tf.expand_dims(tf.convert_to_tensor(cv2.resize(image, target_size), dtype=tf.float32), 0)/255

       # Convert the image to a float32 tensor
    image_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)

    # Split the image into color channels
    r, g, b = tf.split(image_tensor, num_or_size_splits=3, axis=-1)

    # Define the sharpening filter
    sharpening_filter = tf.constant([[-1, -1, -1],
                                     [-1,  9, -1],
                                     [-1, -1, -1]], dtype=tf.float32)*sharpen_factor

    sharpening_filter = tf.reshape(sharpening_filter, [3, 3, 1, 1])

    # Apply the convolution operation to each channel
    r = tf.nn.conv2d(r, filters=sharpening_filter, strides=1, padding='SAME')
    g = tf.nn.conv2d(g, filters=sharpening_filter, strides=1, padding='SAME')
    b = tf.nn.conv2d(b, filters=sharpening_filter, strides=1, padding='SAME')

    # Stack the channels back together
    sharpened_image = tf.concat([r, g, b], axis=-1)
   
    # Remove batch dimension
    sharpened_image = tf.squeeze(sharpened_image, axis=0)
    
    # Clip values to be in the valid image range
    sharpened_image = tf.clip_by_value(sharpened_image, 0.0, 1.0)

    # Convert the tensor back to a NumPy array
    sharpened_image = tf.cast(sharpened_image*255, tf.uint8).numpy()

    # Save the enlarged and sharpened image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, cv2.cvtColor(sharpened_image, cv2.COLOR_RGB2BGR))

# Replace 'input_folder' with the path to the folder containing your images
input_folder = 'Gan_Tut/images/birs_64_images_sorted'
output_folder = 'Gan_Tut/images/birs_64_images_sorted/upscaled'

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        enlarge_and_sharpen(image_path, output_folder)

print("Enlarging and sharpening complete.")
