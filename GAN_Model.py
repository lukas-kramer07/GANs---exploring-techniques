'''
GAN as a custom model subclass with custom training step
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from keras import layers
from keras.losses import BinaryCrossentropy

gan_dir = "num_28"
LATENT_DIM = 100
EPOCHS = 3000
num_examples_to_generate = 16
BATCH_SIZE = 512

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(LATENT_DIM,)))
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

class GAN_Model(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_loss, d_loss, g_opt, d_opt):
        super().compile()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt = g_opt
        self.d_opt = d_opt
    
    #1 equals real, 0 equals fake
    def train_step(self, batch):
        real_images = batch
        
        # meassure gradients of generator and discriminator
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(tf.random.normal([tf.shape(batch)[0], self.latent_dim]), training=True)
            disc_real = self.discriminator(real_images, training=True)
            disc_fake = self.discriminator(fake_images, training=True)

            # Calculate discriminator loss
            y_hat = tf.concat([disc_real, disc_fake], axis=0)
            y = tf.concat([tf.ones_like(disc_real), tf.zeros_like(disc_fake)], axis = 0)
            
            # apply noise to real labels
            y += 0.05*tf.random.normal(tf.shape(y))
            disc_loss = self.d_loss(y, y_hat)

            # Calculate Generator loss
            gen_loss = self.g_loss(tf.ones_like(disc_fake), disc_fake)

        # Calculate and apply gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.d_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return {"d_loss":disc_loss, "g_loss":gen_loss}

    

class ModelMonitor(tf.keras.callbacks.Callback):
    def __init__(self, test_input, checkpoint, checkpoint_prefix):
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint = checkpoint
        self.test_input = test_input
    def on_epoch_end(self, epoch, logs=None):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        if (epoch +1) % 10 == 0:
            predictions = self.model.generator(self.test_input, training=False)
            _ = plt.figure(figsize=(4, 4))

            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.imshow(tf.cast(predictions[i, :, :, 0] * 127.5 +127.5, tf.dtypes.int16), cmap='gray')
                plt.axis("off")
            os.makedirs(
                f"Gan_Tut/plots/{gan_dir}", exist_ok=True
            )  # Create the "models" folder if it doesn't exist
            plt.savefig(f"Gan_Tut/plots/{gan_dir}/image_at_epoch_{epoch+1}.png")
            plt.close()

            #save checkpoints
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

def normalize(element):
    image = element['image']
    return tf.cast((tf.image.resize(image, (28, 28))-127.5) / 127.5, tf.dtypes.float32)

def main():
    
    train_dataset, info = tfds.load("mnist", split="train", with_info=True)

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

    # build generator and discriminator
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator_loss = BinaryCrossentropy()
    discriminator_loss = BinaryCrossentropy()

    # establish checkpoints
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    
    # initiate and compile model
    GAN = GAN_Model(generator=generator, discriminator=discriminator, latent_dim=LATENT_DIM)
    GAN.compile(g_loss=generator_loss, d_loss=discriminator_loss, g_opt=generator_optimizer, d_opt=discriminator_optimizer)

    seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])
    monitor = ModelMonitor(seed, checkpoint, checkpoint_prefix)
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('starte Training')
    history = GAN.fit(train_dataset, epochs=EPOCHS, callbacks=[monitor])


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