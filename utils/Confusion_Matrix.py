'''
makes a confusion-matrix based on the models predictions in crossreference to the true labels
'''
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
def make_confusion_matrix(y_true, y_pred, model_name, classes=None, figsize=(10,10,), text_size=15):
  #Create the confusion matrix
  cm = confusion_matrix(y_true, np.round(y_pred))
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes= cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)
  #Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Purples)
  fig.colorbar(cax)

#set labels to be classes
  labels = classes if classes else np.arange(cm.shape[0])
  # Label the axes
  ax.set(title=f"Confusion matrix - Fashion_model{model_name}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set x-labels to the bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  #Adjust label size
  ax.xaxis.label.set_size(text_size)
  ax.yaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  # Set coluor threshhold
  threshold = (cm.max() + cm.min())/2.

  #Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i,j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)