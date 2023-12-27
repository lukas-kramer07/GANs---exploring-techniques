import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build a simple convolutional neural network
model = models.Sequential()
model.add(layers.Input((28,28,1)))
model.add(layers.Flatten())
model.add(layers.Dense(150, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Set up the TensorBoard profiler callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logsi", histogram_freq=1, profile_batch="50,60"
)

# Train the model with the profiler callback
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=32,
    validation_data=(test_images, test_labels),
    callbacks=[tensorboard_callback],
)

# Save the model in the native Keras format
model.save("mnist_model", save_format="tf")
