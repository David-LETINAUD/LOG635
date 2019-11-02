import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
#import idx2numpy
from plot_fmnist import *
from Neural_network import *
import pickle
import cv2
from sklearn.metrics import f1_score, accuracy_score, recall_score, recall_score,confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time


# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


class_names = [ "Circle2", "Circle5", "Diamond2","Diamond5", "Hexagon2", "Hexagon5", "Triangle2", "Triangle5"]


# Fonction inspirée de : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #tmp = unique_labels(y_true, y_pred)
    #classes = classes[tmp]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label',
           xlim = (-0.5,7.5),
           ylim = (-0.5,7.5)
    )
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

# Lecture X et y
X = np.array(pickle.load( open("X.pickle", 'rb') ))
y = np.array(pickle.load( open("y.pickle", 'rb') ))

# Taille du dataset
train_ratio = 0.8
data_size = len(y)
training_size = int( train_ratio * data_size)

# Resize pictures
WIDTH = 14
HEIGHT = 14
X = np.array([cv2.resize(img, (WIDTH, HEIGHT)) for img in X])

# Mise en forme des données
X = X.reshape(len(y), X.shape[1] * X.shape[2] )/255 # Flatten the array & normalise features
y = np.unique(y, return_inverse=True)[1]            # Conversion des labels en chiffres

# Split les données
X_train = X[:training_size]
y_train = y[:training_size]

X_test = X[training_size:]
y_test = y[training_size:]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#N_FEATURES = 28 * 28 # 28x28 pixels for the images
N_FEATURES = X.shape[1]
N_CLASSES = len(class_names)

start = time.time()

nn = NNClassifier(
    n_classes=N_CLASSES, 
    n_features=N_FEATURES,
    n_hidden_units=100,     # nombre de neurones dans la couche : more is better
    epochs=500,             # +epochs est grand mieux est la précision mais + long est la convergence : more is better
    learning_rate=0.001,   # 0.0005 => 87% d'accuracy sur le test
    n_batches=500,
).train(X_train, y_train);

end = time.time()
print(end - start)

def plot_error(model):
    plt.plot(range(len(model.error_)), model.error_)
    plt.ylabel('Errors')
    plt.xlabel('Epochs')
    plt.show()

plot_error(nn)


y_hat = nn.predict_proba(X_test)

print('Train Accuracy: %.2f%%' % (nn.score(X_train, y_train) * 100))
print('Test Accuracy: %.2f%%' % (nn.score(X_test, y_test) * 100))

y_hat = np.argmax(y_hat, axis = 1)  # Reshape probas vector TO number of the max proba

plot_confusion_matrix(y_hat, y_test,class_names)

def perf_mesure(y_hat, y_test):
    f1 = f1_score(y_hat, y_test, average='weighted')
    acc = accuracy_score(y_hat, y_test)
    rec = recall_score(y_hat, y_test, average='weighted')  
    return [acc,rec, f1]

perf = perf_mesure(y_hat, y_test)
print(perf)