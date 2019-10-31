import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
#import idx2numpy
from plot_fmnist import *
from Neural_network import *
import pickle
import cv2
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import time

# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

class_names = [ "Circle2", "Circle5", "Diamond2","Diamond5", "Hexagon2", "Hexagon5", "Triangle2", "Triangle5"]


########################################  Functions
def perf_mesure(y_hat, y_test):
    f1 = f1_score(y_hat, y_test, average='weighted')
    acc = accuracy_score(y_hat, y_test)
    rec = recall_score(y_hat, y_test, average='weighted')  
    return [acc,rec, f1]

def plot_product_grid(X, y, rows, cols):
    plt.figure(figsize=(2*rows, 2*cols))  # set the size of the figure 10 inches by 10 inches
    for i in range(rows*cols):  # iterates over 25 images
        plt.subplot(rows, cols, i+1)  # indicate each cell in the plot
        plt.xticks([])  # void x axis ticks
        plt.yticks([])  # void y axis ticks
        plt.grid(False)
        plt.imshow(X[i].reshape(28,28), cmap=plt.cm.binary)  # set image to display in current cell
        plt.xlabel(class_names[y[i]])  # class to display in currente cell
    plt.show()

def plot_perf(perf, hyperParam_range, delays,title  ):
    delays = np.array(delays).transpose(1,0)

    fig, axs = plt.subplots(1,2)
    plt.suptitle(title, fontsize=16)
    
    axs[0].title.set_text('Peformances du modèle sur dataset de test')
    axs[0].plot(hyperParam_range, perf, 'x--')
    axs[0].set_ylabel('Accuracy Recall and F1-score')
    axs[0].set_xlabel('Hyperparameter')
    axs[0].legend(['Précision', 'Rappel', 'F1_score'])
    
    axs[1].title.set_text('Temps de calculs')
    axs[1].plot(hyperParam_range, delays, 'x--')
    axs[1].set_ylabel('Delays (s)')
    axs[1].set_xlabel('Hyperparameter')
    axs[1].legend(['predicting_delay','testing_delay'])
    plt.show()

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
# On étudieras ici : n_hidden_units(/nombre de couches cachés)
# training_delay_RNN = []
# predicting_delay_RNN = []
# perf_RNN = []
# best_index_RNN = 0

# #l_rate_range = np.arange(0.0001,0.04,0.0005) #A garder
# # l_rate_range = np.logspace(0.0001, 0.004, 1, endpoint=False)
# l_rate_range = [0.000001,0.00005, 0.0005, 0.001, 0.01, 0.02, 0.03, 0.05]
# #l_rate_range = np.arange(0.002,0.04,0.002) #A garder
# #l_rate_range = np.arange(0.4,1,0.2)
# cpt = 0
# best_accuracy_RNN = 0
# for l_rate in l_rate_range:
#     nn = NNClassifier(
#         n_classes=N_CLASSES, 
#         n_features=N_FEATURES,
#         n_hidden_units=100,     # nombre de neurones dans la couche : more is better
#         epochs=500,             # +epochs est grand mieux est la précision mais + long est la convergence : more is better
#         learning_rate=l_rate,   # 0.0005 => 87% d'accuracy sur le test
#         n_batches=25,
#     )
#     #### Apprentissage
#     start = time.time()
#     nn.train(X_train, y_train)
#     end = time.time()
#     training_delay_RNN.append(end - start)

#     #### Prédiction
#     start = time.time()
#     y_hat = nn.predict_proba(X_test)    
#     end = time.time()
#     predicting_delay_RNN.append(end - start)

#     # Calcul Perfs
#     y_hat = np.argmax(y_hat, axis = 1)  # Reshape probas vector TO number of the max proba
#     perf = perf_mesure(y_hat, y_test)
#     perf_RNN.append(perf)

#     if perf[0]> best_accuracy_RNN:
#         best_accuracy_RNN = perf[0]
#         best_index_RNN = cpt
#     cpt+=1
#     print("l_rate : ",l_rate, "perf : ", perf)

# # Best Perf :
# print("Best accuracy : {} for learning_rate = {}".format(perf_RNN[best_index_RNN][0] , l_rate_range[best_index_RNN] ) )
# print("Learning delay : {} | predicting delay = {}".format(training_delay_RNN[best_index_RNN] , predicting_delay_RNN[best_index_RNN] ) )

# plot_perf(perf_RNN,l_rate_range,[training_delay_RNN,predicting_delay_RNN], "RNN : Hyperparameter = learning rate")


#################### KNN
# HyperParameters : K
training_delay_KNN = []
predicting_delay_KNN = []
perf_KNN = []
best_index_KNN = 0
K_range = range(1,15,2)
best_accuracy_KNN = 0

cpt=0
for k in K_range :
    knn = KNeighborsClassifier(n_neighbors=k)

    #### Apprentissage
    start = time.time()
    knn.fit(X_train, y_train)
    end = time.time()
    training_delay_KNN.append(end-start)

    #### Prédiction
    start = time.time()
    y_pred = knn.predict(X_test)
    end = time.time()
    predicting_delay_KNN.append(end-start)

    perf = perf_mesure(y_pred,y_test)
    perf_KNN.append(perf)

    if perf[0]> best_accuracy_KNN:
        best_accuracy_KNN = perf[0]
        best_index_KNN = cpt
    cpt+=1
    print("K : ",k, "perf : ", perf)

# Best Perf :
print("Best accuracy : {} for learning_rate = {}".format(perf_KNN[best_index_KNN][0] , K_range[best_index_KNN] ) )
print("Learning delay : {} | predicting delay = {}".format(training_delay_KNN[best_index_KNN] , predicting_delay_KNN[best_index_KNN] ) )

plot_perf(perf_KNN,K_range,[training_delay_KNN,predicting_delay_KNN], "KNN : Hyperparameter = K")


#################### SVM
# HyperParameters : Kernel


#################### RN
# HyperParameters : n_hidden_units/n_hidden_units/nombre de couches cachés


#################### decision tree
# HyperParameters : profondeur