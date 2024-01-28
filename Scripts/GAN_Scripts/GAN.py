"""
GAN for the MNIST dataset
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from keras import layers
import time

gan_dir = "celeb_64"


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 28, 28, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="tanh",
        )
    )
    assert model.output_shape == (None, 28, 28, 1)
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="valid", input_shape=[28, 28, 1]
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="valid"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

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

        # Produce images every 100 epochs as you go
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed, dataset)

        # Save the model every 100 epochs
        if (epoch + 1) % 10 == 0:
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
    _ = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow(tf.cast(predictions[i, :, :, 0] * 127.5 +127.5, tf.dtypes.int16), cmap='gray')
                plt.axis("off")
    os.makedirs(
        f"Training/plots/{gan_dir}", exist_ok=True
    )  # Create the "models" folder if it doesn't exist
    plt.savefig(f"Training/plots/{gan_dir}/image_at_epoch_{epoch}.png")
    plt.close()


# -------------------------------------------------------------------------------------------------------


def normalize(element):
    image = element['image']
    return tf.cast((tf.image.resize(image, (28, 28))-127.5) / 127.5, tf.dtypes.float32)



def main():
    
    train_dataset, info = tfds.load("mnist", split="train", with_info=True)
    BATCH_SIZE = 64
    BUFFER_SIZE = info.splits['train'].num_examples
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = (
        train_dataset
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint_dir = "Training/training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('starte Training')
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
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    main()
