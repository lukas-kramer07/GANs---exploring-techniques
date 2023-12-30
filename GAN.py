"""
GAN for the MNIST dataset
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import layers
import time

gan_dir = "Cifar10_32"


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            3,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="sigmoid",
        )
    )
    assert model.output_shape == (None, 32, 32, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="valid", input_shape=[32, 32, 3]
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="valid"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="valid"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(
    images,
    generator,
    discriminator,
    BATCH_SIZE,
    noise_dim,
    generator_optimizer,
    discriminator_optimizer,
):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def train(
    dataset,
    epochs,
    generator,
    discriminator,
    BATCH_SIZE,
    noise_dim,
    generator_optimizer,
    discriminator_optimizer,
    seed,
    checkpoint,
    checkpoint_prefix,
):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(
                image_batch,
                generator,
                discriminator,
                BATCH_SIZE,
                noise_dim,
                generator_optimizer,
                discriminator_optimizer,
            )

        # Produce images every 10 epochs as you go
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed, dataset)

        # Save the model every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed, dataset)


# ----------------------------------------------------------------------------------------------------------------------------------
def generate_and_save_images(model, epoch, test_input, dataset):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    print(predictions.shape)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0] // 2):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :] * 255)
        plt.axis("off")
    for i in range(8, 16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.cast(next(iter(dataset))[i] * 255, tf.dtypes.int16))
        plt.axis("off")
    os.makedirs(
        f"Gan_Tut/plots/{gan_dir}", exist_ok=True
    )  # Create the "models" folder if it doesn't exist
    plt.savefig(f"Gan_Tut/plots/{gan_dir}/image_at_epoch_{epoch}.png")
    plt.close()


# -------------------------------------------------------------------------------------------------------
def main():
    ds, info = tfds.load("cifar10", split="train", with_info=True)
    train_images = np.array([x["image"] for x in ds if x["label"] == 0])
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype(
        "float32"
    )
    train_images = train_images / 255  # Normalize the images to [-1, 1]
    BUFFER_SIZE = 60000
    BATCH_SIZE = 512
    # Batch and shuffle the data
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    EPOCHS = 2500
    noise_dim = 100
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    train(
        train_dataset,
        EPOCHS,
        generator,
        discriminator,
        BATCH_SIZE,
        noise_dim,
        generator_optimizer,
        discriminator_optimizer,
        seed,
        checkpoint,
        checkpoint_prefix,
    )


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    print(gpus)
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    main()
