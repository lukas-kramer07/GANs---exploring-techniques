import tensorflow as tf
import keras
# Define the input label
input_label = tf.constant([0], dtype=tf.int32)

# Define the size of the embedding space
embedding_dim = 4


# Apply the embedding layer to the input label
embedded_output = keras.layers.Embedding(10, 50)(input_label)

# Print the original label and the corresponding embedded representation
print("Original Label:")
print(input_label.numpy())
print("\nEmbedded Representation:")
print(embedded_output.numpy())
