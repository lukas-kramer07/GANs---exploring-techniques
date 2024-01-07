'''
GAN as a custom model subclass with custom training step
'''


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from keras import layers

gan_dir = "celeb_64"
LATENT_DIM = 100
EPOCHS = 3000
num_examples_to_generate = 16


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 64, 64, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            3,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            activation="sigmoid",
        )
    )
    assert model.output_shape == (None, 64, 64, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="valid", input_shape=[64, 64, 3]
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="valid"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="valid"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

class GAN_Model(tf.keras.model):
    def __init__(self, generator, discriminator, latent_dim):
        super.__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_loss, d_loss, g_opt, d_opt):
        
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt = g_opt
        self.d_opt = d_opt
    
    #1 equals real, 0 equals fake
    def train_step(self, batch):
        real_images = batch
        fake_images = self.generator(tf.random.normal([self.batch_size, self.latent_dim]), training=True)

        # meassure gradients of generator and discriminator
        with tf.GradientTape() as gen_tape, tf.GradientTape as disc_tape:
            disc_real = self.discriminator(real_images, training=True)
            disc_fake = self.discriminator(fake_images, training=True)

            # Calculate discriminator loss
            y_hat = tf.concat(disc_real, disc_fake)
            y = tf.concat(tf.ones_like(disc_real), tf.zeros_like(disc_fake))
            
            # apply noise to real labels
            # y += 0.05*tf.random.normal(tf.shape(y))
            disc_loss = self.d_loss(y, y_hat)

            # Calculate Generator loss
            gen_loss = self.g_loss(tf.ones_like(disc_fake), disc_fake)

        # Calculate and apply gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradients(disc_loss, self.discriminator.trainable_variables)

        self.g_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.d_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return {"d_loss":disc_loss, "g_loss":gen_loss}
    

class ModelMonitor(tf.keras.Callback):
    def __init__(self, test_input):
        self.test_input = test_input
    def on_epoch_end(self, epoch, logs=None):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.model.generator(self.test_input, training=False)
        print(predictions.shape)
        _ = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(tf.cast(predictions[i, :, :, :] * 255, tf.dtypes.int16))
            plt.axis("off")
        os.makedirs(
            f"Gan_Tut/plots/{gan_dir}", exist_ok=True
        )  # Create the "models" folder if it doesn't exist
        plt.savefig(f"Gan_Tut/plots/{gan_dir}/image_at_epoch_{epoch}.png")
        plt.close()

def normalize(element):
    image = element['image']
    return tf.cast(tf.image.resize(image, (64, 64)) / 255, tf.dtypes.float32)

def main():
    
    train_dataset, info = tfds.load("celeb_a", split="train", with_info=True)
    BATCH_SIZE = 800
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
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])
    monitor = ModelMonitor(seed)
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('starte Training')
    # history = TODO


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