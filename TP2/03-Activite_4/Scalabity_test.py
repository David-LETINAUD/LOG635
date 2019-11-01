import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
#import idx2numpy
from plot_fmnist import *
from Neural_network import *
import pickle
import cv2
from sklearn.metrics import f1_score, accuracy_score, recall_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
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

def plot_perf(perf, hyperParam_range, delays,title  ):
    #delays = np.array(delays).transpose(1,0)

    fig, axs = plt.subplots(1,3)
    plt.suptitle(title, fontsize=16)
    
    axs[0].title.set_text('Peformances du modèle sur dataset de test')
    axs[0].plot(hyperParam_range, perf, 'x--')
    axs[0].set_ylabel('Accuracy Recall and F1-score')
    axs[0].set_xlabel('Training ratio')
    axs[0].legend(['Précision', 'Rappel', 'F1_score'])
    
    axs[1].title.set_text('Temps d\'apprentissage')
    axs[1].plot(hyperParam_range, delays[0], 'x--')
    axs[1].set_ylabel('Delays (s)')
    axs[1].set_xlabel('Training ratio')
    #axs[1].legend(['predicting_delay'])

    axs[2].title.set_text('Temps de prédiction')
    axs[2].plot(hyperParam_range, delays[1], 'x--')
    axs[2].set_ylabel('Delays (s)')
    axs[2].set_xlabel('Training ratio')
    #axs[2].legend(['testing_delay'])
    plt.show()


########################################  Load & Data manipulations
# Lecture X et y
X = np.array(pickle.load( open("X2.pickle", 'rb') ))
y = np.array(pickle.load( open("y2.pickle", 'rb') ))

# Taille du dataset
data_size = len(y)

# Resize pictures
WIDTH = 14
HEIGHT = 14
X = np.array([cv2.resize(img, (WIDTH, HEIGHT)) for img in X])

# Mise en forme des données
X = X.reshape(len(y), X.shape[1] * X.shape[2] )/255 # Flatten the array & normalise features
y = np.unique(y, return_inverse=True)[1]            # Conversion des labels en chiffres


#N_FEATURES = 28 * 28 # 28x28 pixels for the images
N_FEATURES = X.shape[1]
N_CLASSES = len(class_names)
ratio_range = [0.2,0.4,0.6,0.8]

########################################  Scalabity tests

#################### Neural Network study
# training_delay_RN = []
# predicting_delay_RN = []
# perf_RN = []
# best_index_RN = 0
# best_y_test_RN =  []

# cpt = 0
# best_accuracy_RN = 0

# nn = NNClassifier(
#         n_classes=N_CLASSES, 
#         n_features=N_FEATURES,
#         n_hidden_units=100,     # nombre de neurones dans la couche : more is better
#         epochs=500,             # +epochs est grand mieux est la précision mais + long est la convergence : more is better
#         learning_rate=0.001,   
#         n_batches=25,
#     )
# for ratio in ratio_range:
    
#     training_size = int( ratio * data_size)
#     # Split les données
#     X_train = X[:training_size]
#     y_train = y[:training_size]

#     X_test = X[training_size:]
#     y_test = y[training_size:]

#     #### Apprentissage
#     start = time.time()
#     nn.train(X_train, y_train)
#     end = time.time()
#     training_delay_RN.append(end - start)

#     #### Prédiction
#     start = time.time()
#     y_hat = nn.predict_proba(X_test)    
#     end = time.time()
#     predicting_delay_RN.append(end - start)

#     # Calcul Perfs
#     y_hat = np.argmax(y_hat, axis = 1)  # Reshape probas vector TO number of the max proba
#     perf = perf_mesure(y_hat, y_test)
#     perf_RN.append(perf)

#     if perf[0]> best_accuracy_RN:
#         best_accuracy_RN = perf[0]
#         best_index_RN = cpt
#         best_y_pred_RN =  y_hat
#     cpt+=1
#     print("ratio : ",ratio, "perf : ", perf)

# # Best Perf :
# print("Best accuracy : {} for ratio = {}".format(perf_RN[best_index_RN][0] , ratio_range[best_index_RN] ) )
# print("Learning delay : {} | predicting delay = {}".format(training_delay_RN[best_index_RN] , predicting_delay_RN[best_index_RN] ) )

# plot_perf(perf_RN,ratio_range,[training_delay_RN,predicting_delay_RN], "RN : Test de scalabilité")

#################### KNN
training_delay_KNN = []
predicting_delay_KNN = []
perf_KNN = []
best_index_KNN = 0
best_accuracy_KNN = 0
best_y_pred_KNN = []

cpt=0
knn = KNeighborsClassifier(n_neighbors=1)
for ratio in ratio_range:
    
    training_size = int( ratio * data_size)
    # Split les données
    X_train = X[:training_size]
    y_train = y[:training_size]

    X_test = X[training_size:]
    y_test = y[training_size:]

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
    print("Ratio : ",ratio, "perf : ", perf)

# Best Perf :
print("Best accuracy : {} for ratio = {}".format(perf_KNN[best_index_KNN][0] , ratio_range[best_index_KNN] ) )
print("Learning delay : {} | predicting delay = {}".format(training_delay_KNN[best_index_KNN] , predicting_delay_KNN[best_index_KNN] ) )

plot_perf(perf_KNN,ratio_range,[training_delay_KNN,predicting_delay_KNN], "KNN : Test de scalabilité")

#################### SVM
# HyperParameters : Kernel


#################### RN
# HyperParameters : n_hidden_units/n_hidden_units/nombre de couches cachés


#################### decision tree
# HyperParameters : profondeur