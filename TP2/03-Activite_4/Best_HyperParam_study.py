import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
#import idx2numpy
from plot_fmnist import *
from Neural_network import *
import pickle
import cv2

# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


class_names = [ "Circle2", "Circle5", "Diamond2","Diamond5", "Hexagon2", "Hexagon5", "Triangle2", "Triangle5"]

########################################  Functions
def accuracy(y_predict,y):
    y_pred_int = np.array( [np.argmax(p) for p in y_predict])
    y_int = np.array( [np.argmax(p) for p in y])
    VP = np.sum( np.equal(y_pred_int, y_int) )
    return VP/len(y_int)

########################################  Load & Data manipulations
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

########################################  HyperParametersStudy

#################### Neural Network study
# HyperParameters : learning_rate/n_hidden_units/epochs/n_batches
# On étudieras ici : n_hidden_units/n_hidden_units (/nombre de couches cachés)


#################### KNN
# HyperParameters : K


#################### SVM
# HyperParameters : Kernel


#################### RN
# HyperParameters : n_hidden_units/n_hidden_units/nombre de couches cachés


#################### decision tree
# HyperParameters : profondeur