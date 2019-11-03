import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.neural_network import MLPClassifier
import pickle

# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#################################  Functions
def flatten_x(x):
    return np.array([i.flatten() for i in x])

def rn_model(x_train, y_train, x_test):
    classifier = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
    classifier.fit(x_train, y_train) 
    y_pred = classifier.predict(x_test)
    return y_pred


# Lecture X et y
X = pickle.load( open("X.pickle", 'rb') )
y = pickle.load( open("y.pickle", 'rb') )

# Taille du dataset
train_ratio = 0.8
data_size = len(y)
training_size = int( train_ratio * data_size)

# Mise en forme des données
X = flatten_x(X)

# Split les données
X_train = X[:training_size]
y_train = y[:training_size]

X_test = X[training_size:]
y_test = y[training_size:]

# Training with X and y
y_pred = rn_model(X_train, y_train, X_test)
print("Done")