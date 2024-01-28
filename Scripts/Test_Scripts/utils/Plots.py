import matplotlib.pyplot as plt
import numpy as np

def change_plot(history):
     # Calculate the change in accuracy from the previous epoch
    accuracy_changes = [0] + [history.history['accuracy'][i] - history.history['accuracy'][i-1] for i in range(1, len(history.history['accuracy']))]

    plt.figure(figsize=(10, 6))

    # Plot accuracy and validation accuracy
    accuracy_line, = plt.plot(history.history['accuracy'], label='accuracy', color='b')
    val_accuracy_line, = plt.plot(history.history['val_accuracy'], label='val_accuracy', color='g')

    # Plot change in accuracy
    accuracy_change_line, = plt.plot(accuracy_changes, label='Accuracy Change', color='r', linestyle='dashed')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])

    if 'lr' in history.history:
      # Create a twin axis for the learning rate
      ax2 = plt.gca().twinx()
      lr_line, = ax2.plot(history.history['lr'], label='Learning Rate', color='m', linestyle='dotted')
      ax2.set_ylabel('Learning Rate')

      # Combine the legend entries from both axes
      lines = [accuracy_line, val_accuracy_line, accuracy_change_line, lr_line]
      labels = [line.get_label() for line in lines]
      plt.legend(lines, labels, loc='upper left')
      plt.title('Accuracy, Validation Accuracy, Accuracy Change, and Learning Rate')
      plt.tight_layout()
    else:
      lines = [accuracy_line, val_accuracy_line, accuracy_change_line]
      labels = [line.get_label() for line in lines]
      plt.legend(lines, labels, loc='upper left')
      plt.title('Accuracy, Validation Accuracy, Accuracy Change')
      plt.tight_layout()