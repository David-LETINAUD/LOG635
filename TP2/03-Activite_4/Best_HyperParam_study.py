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
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import sklearn.metrics as metrics

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
    axs[0].set_xlabel('Hyperparameter')
    axs[0].legend(['Précision', 'Rappel', 'F1_score'])
    
    axs[1].title.set_text('Temps d\'apprentissage')
    axs[1].plot(hyperParam_range, delays[0], 'x--')
    axs[1].set_ylabel('Delays (s)')
    axs[1].set_xlabel('Hyperparameter')
    #axs[1].legend(['predicting_delay'])

    axs[2].title.set_text('Temps de prédiction')
    axs[2].plot(hyperParam_range, delays[1], 'x--')
    axs[2].set_ylabel('Delays (s)')
    axs[2].set_xlabel('Hyperparameter')
    #axs[2].legend(['testing_delay'])
    plt.show()


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

########################################  Load & Data manipulations
# Lecture X et y
X = np.array(pickle.load( open("X.pickle", 'rb') ))
y = np.array(pickle.load( open("y.pickle", 'rb') ))

# Taille du dataset
train_ratio = 0.8
data_size = len(y)
print(data_size)
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
# training_delay_RN = []
# predicting_delay_RN = []
# perf_RN = []
# best_index_RN = 0
# best_y_test_RN =  []

# #l_rate_range = np.arange(0.0001,0.04,0.0005) #A garder
# # l_rate_range = np.logspace(0.0001, 0.004, 1, endpoint=False)
# #l_rate_range = [0.000001,0.00005, 0.0005, 0.001, 0.01, 0.02, 0.03, 0.05]
# l_rate_range = [0.000001,0.00005, 0.0005,0.0008, 0.001,0.003, 0.005, 0.01,0.012]
# #l_rate_range = np.arange(0.002,0.04,0.002) #A garder
# #l_rate_range = np.arange(0.4,1,0.2)
# cpt = 0
# best_accuracy_RN = 0
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
#     print("l_rate : ",l_rate, "perf : ", perf)

# # Best Perf :
# print("Best accuracy : {} for learning_rate = {}".format(perf_RN[best_index_RN][0] , l_rate_range[best_index_RN] ) )
# print("Learning delay : {} | predicting delay = {}".format(training_delay_RN[best_index_RN] , predicting_delay_RN[best_index_RN] ) )

# plot_perf(perf_RN,l_rate_range,[training_delay_RN,predicting_delay_RN], "RN : Hyperparameter = learning rate")
# plot_confusion_matrix(y_test,best_y_pred_RN,class_names)


#################### KNN
# HyperParameters : K
training_delay_KNN = []
predicting_delay_KNN = []
perf_KNN = []
best_index_KNN = 0
K_range = range(1,15,2)
best_accuracy_KNN = 0
best_y_pred_KNN = []

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
        best_y_pred_KNN = y_pred
    cpt+=1
    print("K : ",k, "perf : ", perf)

# Best Perf :
print("Best accuracy : {} for learning_rate = {}".format(perf_KNN[best_index_KNN][0] , K_range[best_index_KNN] ) )
print("Learning delay : {} | predicting delay = {}".format(training_delay_KNN[best_index_KNN] , predicting_delay_KNN[best_index_KNN] ) )

plot_perf(perf_KNN,K_range,[training_delay_KNN,predicting_delay_KNN], "KNN : Hyperparameter = K")
plot_confusion_matrix(y_test,best_y_pred_KNN,class_names)

#################### SVM
# HyperParameters : Kernel
training_delay_svm = []
predicting_delay_svm = []
perf_svm = []
best_index_svm = 0

clf_svm = svm.SVC(gamma='scale')

start = time.time()
clf_svm.fit(X_train, y_train) 
end = time.time()
training_delay_svm.append(end-start)

start = time.time()
y_pred = clf_svm.predict(X_test)
end = time.time()
predicting_delay_svm.append(end-start)

perf = perf_mesure(y_pred,y_test)
perf_svm.append(perf)

# Best Perf :
print("Best accuracy : {} for learning_rate = {}".format(perf_svm[best_index_svm][0] , K_range[best_index_svm] ) )
print("Learning delay : {} | predicting delay = {}".format(training_delay_svm[best_index_svm] , predicting_delay_svm[best_index_svm] ) )

plot_perf(perf_svm,range(1,2),[training_delay_svm,predicting_delay_svm], "SVM: Hyperparameter: Kernel?")
plot_confusion_matrix(y_test,y_pred,class_names)

#################### RN
# HyperParameters : n_hidden_units/nombre de couches cachés
training_delay_rn = []
predicting_delay_rn = []
perf_rn = []
best_index_rn = 0
best_accuracy_rn = 0

classifier = MLPClassifier(solver='lbfgs', learning_rate_init=0.01, hidden_layer_sizes=(30,), random_state=1)

start = time.time()
classifier.fit(X_train, y_train) 
end = time.time()
training_delay_rn.append(end-start)

start = time.time()
y_pred = classifier.predict(X_test)
end = time.time()
predicting_delay_rn.append(end-start)

perf = perf_mesure(y_pred,y_test)
perf_rn.append(perf)

print("Best accuracy : {} for learning_rate = {}".format(perf_rn[best_index_rn][0] , K_range[best_index_rn] ) )
print("Learning delay : {} | predicting delay = {}".format(training_delay_rn[best_index_rn] , predicting_delay_rn[best_index_rn] ) )

plot_perf(perf_rn,range(1,2),[training_delay_rn,predicting_delay_rn], "RN: Hyperparameter = ")
plot_confusion_matrix(y_test,y_pred,class_names)

#################### decision tree
# HyperParameters : profondeur
training_delay_Tree = []
predicting_delay_Tree = []
perf_Tree = []
best_index_Tree = 0
best_accuracy_Tree = 0
best_y_pred_Tree = []

clf = tree.DecisionTreeClassifier()

start = time.time()
clf = clf.fit(X_train,y_train)
end = time.time()
training_delay_Tree.append(end-start)

start = time.time()
Y_pred = clf.predict(X_test)
end = time.time()
predicting_delay_Tree.append(end-start)

perf = perf_mesure(y_pred,y_test)
perf_Tree.append(perf)

accuracy = metrics.accuracy_score(y_test, Y_pred)

# Best Perf :
print("Tree accuracy: ", accuracy)
print("Learning delay : {} | predicting delay = {}".format(training_delay_Tree[0] , predicting_delay_Tree[0] ) )

plot_perf(perf_Tree,range(1,2),[training_delay_Tree,predicting_delay_Tree], "Tree : Hyperparameter = profondeur")
plot_confusion_matrix(y_test,y_pred,class_names)
