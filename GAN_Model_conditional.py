'''
GAN as a custom model subclass with custom training step
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from keras import layers, Model
from keras.losses import BinaryCrossentropy
import numpy as np

gan_dir = "num_28"
LATENT_DIM = 128
EPOCHS = 10000
num_examples_to_generate = 16
BATCH_SIZE = 512

def make_generator_model(latent_dim=LATENT_DIM, classes=10):

    input_latent = layers.Input(shape=latent_dim)
    lat= layers.Dense(7*7*128, use_bias=False)(input_latent)
    lat= layers.BatchNormalization()(lat)
    lat= layers.LeakyReLU()(lat)
    lat= layers.Reshape((7, 7, 128))(lat)

    input_label = layers.Input(shape=(1,))
    il = layers.Embedding(classes, 50)(input_label)
    il = layers.Dense(7*7)(il)
    il = layers.Reshape((7, 7,1))(il)

    merge = layers.Concatenate()([lat, il])
    #assert merge.shape == (None, 7, 7, 129)  # Note: None is the batch size

    x= layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merge)
    #assert x.shape == (None, 7, 7, 128)
    x= layers.BatchNormalization()
    x= layers.LeakyReLU()

    x= layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    #assert x.shape == (None, 14, 14, 128)
    x= layers.BatchNormalization()(x)
    x= layers.LeakyReLU()(x)

    output= layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    model = Model([input_latent, input_label], output)
    return model


def make_discriminator_model(in_shape = (28,28,1), classes=10):
    input_label = layers.Input(shape=(1,))
    il = layers.Embedding(classes, 50)(input_label)
    il = layers.Dense(in_shape[0]*in_shape[1])(il)
    il = layers.Reshape((in_shape[0], in_shape[1],1))(il)

    input_image = layers.Input(shape=in_shape)   

    merge = layers.Concatenate()([input_image, il])

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merge)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    model = Model([input_image, input_label], output)
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
        (real_images, labels) = batch
        
        # meassure gradients of generator and discriminator
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(tf.random.normal([[tf.shape(batch)[0], self.latent_dim], labels]), training=True)
            disc_real = self.discriminator([real_images, labels], training=True)
            disc_fake = self.discriminator([fake_images, labels], training=True)

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
    def __init__(self, test_input, test_labels, checkpoint, checkpoint_prefix):
        self.labels = test_labels
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint = checkpoint
        self.test_input = test_input
    def on_epoch_end(self, epoch, logs=None):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        if (epoch +1) % 10 == 0:
            predictions = self.model.generator(self.test_input, self.labels, training=False)
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
    image,label = element['image'], element['label']
    return tf.cast((tf.image.resize(image, (28, 28))-127.5) / 127.5, tf.dtypes.float32), label

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
    test_labels = np.random.randint(0, 10, size=(16, 1))
    monitor = ModelMonitor(seed,test_labels, checkpoint, checkpoint_prefix)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
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