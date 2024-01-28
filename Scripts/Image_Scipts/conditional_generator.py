import tensorflow as tf
import keras
import matplotlib.pyplot as plt

LATENT_DIM = 100
gan_dir = "fashion_28"
num_examples_to_generate = 16
classes = {
    'T-shirt/Top' : 0,
    'Trouser' : 1,
    'Pullover' : 2,
    'Dress' : 3,
    'Coat' : 4,
    'Sandal' : 5,
    'Shirt' : 6,
    'Sneaker' : 7,
    'Bag' : 8,
    'Ankle boot' : 9
}
def predict(label, seed):
    labels = tf.expand_dims(tf.convert_to_tensor([label] * num_examples_to_generate), axis=-1)
    generator = keras.models.load_model(f'Training/training_checkpoints/{gan_dir}/model.keras')
    predictions = generator([seed, labels], training=False)
    _ = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.cast(predictions[i, :, :, 0] * 127.5 +127.5, tf.dtypes.int16), cmap='gray')
        plt.axis("off")
    plt.show()

def main():
    seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])
    user_input = input(f'select a class or a label from this list: \n{classes}')
    if user_input.isdigit():
        if user_input in classes:
            predict(classes[user_input], seed)
        elif int(user_input) in classes.values():
            predict(int(user_input), seed)
        else: 
            print('error')
    else:
        if user_input in classes: 
            predict(classes[user_input], seed)
        else:
            print('error')
    again = input('try again?')
    if  again.lower() == 'true' or again == '1' or again.lower() == 'yes' or again.lower() == 'y':
        main()

if __name__ == "__main__":
    main()