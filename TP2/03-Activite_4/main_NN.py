import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
#import idx2numpy
from plot_fmnist import *
from Neural_network import *
import pickle
import cv2
from sklearn.metrics import f1_score, accuracy_score, recall_score
import time


# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


class_names = [ "Circle2", "Circle5", "Diamond2","Diamond5", "Hexagon2", "Hexagon5", "Triangle2", "Triangle5"]


# Lecture X et y
X = np.array(pickle.load( open("X.pickle", 'rb') ))
y = np.array(pickle.load( open("y.pickle", 'rb') ))

# Taille du dataset
train_ratio = 0.8
data_size = len(y)
training_size = int( train_ratio * data_size)

# Resize pictures
WIDTH = 28
HEIGHT = 28
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
    learning_rate=0.0005,   # 0.0005 => 87% d'accuracy sur le test
    n_batches=25,
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

def perf_mesure(y_hat, y_test):
    f1 = f1_score(y_hat, y_test, average='weighted')
    acc = accuracy_score(y_hat, y_test)
    rec = recall_score(y_hat, y_test, average='weighted')  
    return [acc,rec, f1]

perf = perf_mesure(y_hat, y_test)
print(perf)