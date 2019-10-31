import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm

LABELS = [ "Circle2", "Circle5", "Diamond2","Diamond5", "Hexagon2", "Hexagon5", "Triangle2", "Triangle5"]


#################################  Functions
def flatten_x(x):
    return np.array([i.flatten() for i in x])

def vectorize_y(y):
    y_vec = []
    for str_lbl in y :
        y_vec.append( [1*(i==str_lbl) for i in LABELS] )
    return np.array(y_vec)

def knn_train(x_train, y_train, x_test, y_test):
    k_range = range(1, 5)
    scores = {}
    scores_list = []
    moyenne = 0
    nb = 0
    for k in k_range:
        nb += 1
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(scores[k])
        moyenne += scores[k]
    return moyenne / nb



# Lecture X et y
X = pickle.load( open("X.pickle", 'rb') )
y = pickle.load( open("y.pickle", 'rb') )

# Taille du dataset
train_ratio = 0.8
data_size = len(y)
training_size = int( train_ratio * data_size)

# Mise en forme des données
X = flatten_x(X)
y = vectorize_y(y) 

# Split les données
X_train = X[:training_size]
y_train = y[:training_size]

X_test = X[training_size:]
y_test = y[training_size:]

# Training with X and y
moyenne = knn_train(X_train, y_train, X_test, y_test)
    
print("RESULTS: " + str(moyenne))