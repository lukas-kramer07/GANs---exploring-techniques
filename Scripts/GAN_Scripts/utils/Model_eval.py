import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import Plots
from utils import Confusion_Matrix
import os

def model_eval(model, model_name, history, test_ds, class_names):
    Plots.change_plot(history)
    # Create the "Training/plots" folder if it doesn't exist
    os.makedirs(f"Training/plots/{model_name}", exist_ok=True)
    plt.savefig(f"Training/plots/{model_name}/history_with_lr_and_change.png")

    y_probs = model.predict(test_ds)
    y_preds = tf.argmax(y_probs, axis=1)
    y_true = np.concatenate([y for x, y in test_ds], axis=0) # extract labels from test_ds
    y_true = tf.argmax(y_true, axis=1) # revert from one_hot
    Confusion_Matrix.make_confusion_matrix(y_true=y_true,
                        y_pred=y_preds,
                        classes=class_names,
                        figsize=(13,13),
                        text_size=8,
                        model_name=model_name)
    os.makedirs(f"Training/plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"Training/plots/{model_name}/confusion_matrix")

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"test_acc: {test_acc}; test_loss: {test_loss}")
    model.summary()
    os.makedirs(f"Training/training_checkpoints", exist_ok=True)  # Create the "models" folder if it doesn't exist
    model.save(f"Training/training_checkpoints/{model_name}")#-> move to utils