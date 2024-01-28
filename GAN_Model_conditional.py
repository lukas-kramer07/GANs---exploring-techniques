'''
GAN as a custom model subclass with custom training step
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import keras
from keras import layers, Model
from keras.constraints import Constraint
import numpy as np
from keras import backend as K

gan_dir = "nums_64"
LATENT_DIM = 100
EPOCHS = 40000
num_examples_to_generate = 20
BATCH_SIZE = 512
ITERATIONS_CRITIC = 5

## Wasserstein specific functions
# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
 return K.mean(y_true * y_pred)

# clip model weights to a bounding area defined by clip_value
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)
    
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


## MODEL definiton
def make_generator_model(latent_dim=LATENT_DIM, classes=5):

    input_latent = layers.Input(shape=latent_dim)
    lat= layers.Dense(8*8*latent_dim, use_bias=False)(input_latent)
    lat= layers.BatchNormalization()(lat)
    lat= layers.LeakyReLU()(lat)
    lat= layers.Reshape((8, 8, latent_dim))(lat)

    input_label = layers.Input(shape=(1,))
    il = layers.Embedding(classes, 50)(input_label)
    il = layers.Dense(8*8)(il)
    il = layers.Reshape((8, 8,1))(il)

    merge = layers.Concatenate()([lat, il])
    #assert merge.shape == (None, 7, 7, 129)  # Note: None is the batch size

    x= layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merge)
    #assert x.shape == (None, 7, 7, 128)
    x= layers.BatchNormalization()(x)
    x= layers.LeakyReLU()(x)

    x= layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    #assert x.shape == (None, 14, 14, 128)
    x= layers.BatchNormalization()(x)
    x= layers.LeakyReLU()(x)

    x= layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    #assert x.shape == (None, 14, 14, 128)
    x= layers.BatchNormalization()(x)
    x= layers.LeakyReLU()(x)

    output= layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)

    model = Model([input_latent, input_label], output)
    return model


def make_critic_model(in_shape = (64,64,1), classes=5):
    const = ClipConstraint(0.01)
    input_label = layers.Input(shape=(1,))
    il = layers.Embedding(classes, 50)(input_label)
    il = layers.Dense(in_shape[0]*in_shape[1])(il)
    il = layers.Reshape((in_shape[0], in_shape[1],1))(il)

    input_image = layers.Input(shape=in_shape)   

    merge = layers.Concatenate()([input_image, il])

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_constraint=const)(merge)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_constraint=const)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    # add linear output 
    output = layers.Dense(1)(x)

    model = Model([input_image, input_label], output)
    return model


class GAN_Model(tf.keras.Model):
    def __init__(self, generator, critic, latent_dim):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim

    def compile(self, g_loss, d_loss, g_opt, d_opt):
        super().compile()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt = g_opt
        self.d_opt = d_opt
    
    # -1 equals real, 1 equals fake; although the critic assigns a score, where higher = more realistic 
    def train_step(self, batch):
        (real_images, labels) = batch
        # meassure gradients of generator and critic
        for _ in range(ITERATIONS_CRITIC):
            with tf.GradientTape() as disc_tape:
                
                fake_images = self.generator([tf.cast(tf.random.normal([tf.shape(labels)[0], self.latent_dim]), dtype=tf.float32), labels], training=True)
                disc_real = self.critic([real_images, labels], training=True)
                disc_fake = self.critic([fake_images, labels], training=True)

                # Calculate critic loss for real images and fake images
                y_real = -1*tf.ones_like(disc_real)
                y_fake = tf.ones_like(disc_fake)

                # apply noise to real labels
                y_real += 0.05*tf.random.normal(tf.shape(y_real))
                y_fake += 0.05*tf.random.normal(tf.shape(y_fake))

                disc_loss_real = self.d_loss(y_real, disc_real)
                disc_loss_fake = self.d_loss(y_fake, disc_fake)

                #combine to obtain loss
                disc_loss = disc_loss_fake + disc_loss_real
            disc_grads = disc_tape.gradient(disc_loss, self.critic.trainable_variables)
            self.d_opt.apply_gradients(zip(disc_grads, self.critic.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_images = self.generator([tf.cast(tf.random.normal([tf.shape(labels)[0], self.latent_dim]), dtype=tf.float32), labels], training=True)
            disc_fake = self.critic([fake_images, labels], training=True)
            # Calculate Generator loss with inverted labels
            gen_loss = self.g_loss(-1*tf.ones_like(disc_fake), disc_fake)

        # Calculate and apply gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        

        self.g_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        

        return {"d_loss_real":disc_loss_real,"d_loss_fake":disc_loss_fake, "g_loss":gen_loss}

    

class ModelMonitor(tf.keras.callbacks.Callback):
    def __init__(self, test_input, test_labels, gan_dir):
        self.gan_dir = gan_dir
        self.labels = test_labels
        self.test_input = test_input
    def on_epoch_end(self, epoch, logs=None):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        if (epoch+1) % 1000  == 0:
            predictions = self.model.generator([self.test_input, self.labels], training=False)
            _ = plt.figure(figsize=(5, 4))

            for i in range(predictions.shape[0]):
                plt.subplot(5, 4, i + 1)
                plt.imshow(tf.cast(predictions[i, :, :, 0] * 127.5 +127.5, tf.dtypes.int16), cmap='gray')
                plt.axis("off")
            os.makedirs(
                f"Gan_Tut/plots/{self.gan_dir}", exist_ok=True
            )  # Create the "models" folder if it doesn't exist
            plt.savefig(f"Gan_Tut/plots/{self.gan_dir}/image_at_epoch_{epoch+1}.png")
            plt.close()

            #save model
            os.makedirs(
                f"training_checkpoints/{self.gan_dir}/", exist_ok=True
            )  # Create the "models" folder if it doesn't exist
            self.model.generator.save(f'training_checkpoints/{self.gan_dir}/model.keras')

## DATA Manipulation
def normalize(element):
    image,label = element['image'], element['label']
    return tf.cast((tf.image.resize(image, (64, 64))-127.5) / 127.5, tf.dtypes.float32), label
def visualize_data(test_ds, ds_info=None):
    num_images_to_display = 15
    plt.figure(figsize=(num_images_to_display, num_images_to_display))
    count = 0
    # Plot test samples
    for i in range(int(np.ceil(num_images_to_display / BATCH_SIZE))):
        image, label = next(iter(test_ds))
        for n in range(min(BATCH_SIZE, num_images_to_display - i * BATCH_SIZE)):
            plt.subplot(
                2 * int(tf.sqrt(float(num_images_to_display))) + 1,
                2 * int(tf.sqrt(float(num_images_to_display))) + 1,
                n + i + 1,
            )
            img = tf.cast(image[n] * 127.5 +127.5, tf.dtypes.int16)
            plt.imshow(img, cmap='gray')
            if ds_info:
                plt.title(
                    f"Test - {ds_info.features['label'].int2str(int(tf.argmax(label[n])))}",
                    fontsize=10,
                )
            plt.axis("off")
            count += 1
    plt.show() 

## MAIN function
def main():
    
    train_dataset,info = tfds.load('mnist', split='train', with_info=True)
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

    # visualize data
    visualize_data(train_dataset)

    # build generator and critic
    generator = make_generator_model()
    critic = make_critic_model()
    generator_optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)
    critic_optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)
    generator_loss = wasserstein_loss
    critic_loss = wasserstein_loss

    
    # initiate and compile model
    GAN = GAN_Model(generator=generator, critic=critic, latent_dim=LATENT_DIM)
    GAN.compile(g_loss=generator_loss, d_loss=critic_loss, g_opt=generator_optimizer, d_opt=critic_optimizer)

    seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])
    test_labels = tf.constant([[0], [0], [0],[0],[1],[1], [1], [1],[2],[2],[2],[2],[3],[3],[3],[3],[4],[4],[4],[4]])#np.random.randint(0, 10, size=(16, 1))
    monitor = ModelMonitor(seed,test_labels, gan_dir)

    print('starte Training')
    history = GAN.fit(train_dataset, epochs=EPOCHS, callbacks=[monitor])
    plt.plot(history.history['d_loss'], label='critic_loss')
    plt.plot(history.history['g_loss'], label='Generator_loss')
    plt.legend()
    plt.savefig(f'Gan_Tut/plots/{gan_dir}/Loss')
    plt.show()

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