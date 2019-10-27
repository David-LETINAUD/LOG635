import numpy as np
import idx2numpy
from tabulate import tabulate
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier

# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


training_set_inputs = np.array([[1,1,0,0,0,0,1],[1,0,0,0,0,0,1],[1,1,0,0,0,1,0],[1,0,0,0,0,1,0],
                               [1,1,0,0,0,1,1],[1,0,0,0,0,1,1],[1,1,0,0,1,0,0],[1,0,0,0,1,0,0],
                               [1,1,0,0,1,0,1],[1,0,0,0,1,0,1],[1,1,0,0,1,1,0],[1,0,0,0,1,1,0],
                               [1,1,0,0,1,1,1],[1,0,0,0,1,1,1],[1,1,0,1,0,0,0],[1,0,0,1,0,0,0],
                               [1,1,0,1,0,0,1],[1,0,0,1,0,0,1],[1,1,0,1,0,1,0],[1,0,0,1,0,1,0],
                               [1,1,0,1,0,1,1],[1,0,0,1,0,1,1],[1,1,0,1,1,0,0],[1,0,0,1,1,0,0],
                               [1,1,0,1,1,0,1],[1,0,0,1,1,0,1],[1,1,0,1,1,1,0],[1,0,0,1,1,1,0],
                               [1,1,0,1,1,1,1],[1,0,0,1,1,1,1],[1,1,1,0,0,0,0],[1,0,1,0,0,0,0],
                               [1,1,1,0,0,0,1],[1,0,1,0,0,0,1],[1,1,1,0,0,1,0],[1,0,1,0,0,1,0],
                               [1,1,1,0,0,1,1],[1,0,1,0,0,1,1],[1,1,1,0,1,0,0],[1,0,1,0,1,0,0]])

# The «.T» function transposes the matrix from horizontal to vertical.
training_set_outputs = np.array([[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]]).T

def sigma(x):
    return 1 / (1 + np.exp(-x))
# def softmax(A):
#     expA = np.exp(A)
#     return expA / expA.sum()

# def softmax(z):
#     z_exp = [np.exp(i) for i in z]
#     sum_z_exp = sum(z_exp)
#     return [i / sum_z_exp for i in z_exp]

# Creating the classifier

def multiClassify():

    sgd_classifier = SGDClassifier(random_state=42)
    # Training
    sgd_classifier.fit(X_train_flat, y_train)
    position = 0

    # Predicted value
    sgd_classifier.predict([X_train_flat[position]])

    # Getting a feature
    some_digit = X_train_flat[0]

    #Acutal value
    y[position]

    # Digit image
    plot_digit(X_train_flat[position])

    #retourne 10 scores
    some_digit_scores = sgd_classifier.decision_function([some_digit])

    #Le score le plus élevé est en effet celui correspondant à la classe prédite
    sgd_classifier.classes_[np.argmax(some_digit_scores)]

    ova_clf = OneVsRestClassifier(sgd_classifier)
    ova_clf.fit(X_train_flat, y_train)
    ova_clf.predict([some_digit])
    #len(ova_clf.estimators_))

class Perceptron:
    """Perceptron classifier."""
    def __init__(self, eta=0.1, n_iters=10):
        self.eta = eta
        self.n_iters = n_iters
 
    def train(self, X, y):
        """Function to train the neuron, which modifies the weights (w) based on the input values 
        and expected results.    
        """
        self.w_ = np.zeros(1 + X.shape[1])#*X.shape[2])
        self.errors_ = []

        for _ in range(self.n_iters):
            errors = 0
            for xi, target in zip(X, y):
                error = target - self.activation(xi)
                update = self.eta * error
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
 
    def activation(self, X):
        """Return class using Heaviside step function
        f(z) = 1 if z >= theta; 0 otherwise.
        """
        #f = np.where(self.predict(X) >= 0.9, 1, 0)
        f = self.predict(X)
        return f
 
    def predict(self, X):
        """Summation function."""
        #z = w · x + theta
        z = sigma(np.dot(X, self.w_[1:]) + self.w_[0])
        #z = softmax(np.dot(X, self.w_[1:]) + self.w_[0])
        return z


def plot_digit(digit):
    digit_image = digit.reshape(28, 28)
    #plt.imshow(digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.imshow(digit_image, cmap = 'gray', interpolation="nearest")
    plt.axis("off")
    plt.show()

# Lecture X et y
X = pickle.load( open("X.pickle", 'rb') )
y = pickle.load( open("y.pickle", 'rb') )

# Découpage du dataset
data_size = len(y)
training_size = int( 0.8 * data_size)

X_train = X[0:training_size]

#X_train_flat = np.array([resize(i, (28, 28)).flatten() for i in X_train])
X_train_flat = np.array([i.flatten() for i in X_train])

#y_train = np.array([y[0:training_size]]).T
y_train = y[0:training_size]

labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_train = np.array([y_train]).T

print(y_train.shape)
# A faire : transformer y = [1,0,0,2,4,8...]    # Chaque numéro correspond à 1 classe (ici 8 classes)
#                EN     y = [[0,1,0,0,0,0,0,0],  # correspond à 1
#                            [1,0,0,0,0,0,0,0],   # correspond à 0
#                            [1,0,0,0,0,0,0,0],
#                            [0,0,1,0,0,0,0,0],   # correspond à 2
#                            ...

multiClassify()


# Autre idée : Faire 1 perceptron pour chaque classe (ici 8 classes)
#              Chaque predict d'1 perceptron sortira une proba en 0 et 1 -> activation prend une décision
#              Combinaison de chaque résultat dans une tableau 1x8 pour correspondre à la décision finale

X_test = X[training_size:data_size]
#X_test_flat = np.array([resize(i, (28, 28)).flatten() for i in X_test])
X_test_flat = np.array([i.flatten() for i in X_test])
y_test = y[training_size:data_size]

y_test = labelencoder_y.fit_transform(y_test)
y_test = np.array([y_test]).T

print(y_test.shape)

print(training_set_inputs.shape,training_set_outputs.shape )
print(data_size, X_train_flat.shape, y_train.shape)
print(data_size, X_test_flat.shape, y_test.shape)

# Init the perceptron
ppn = Perceptron(eta=0.1, n_iters=100)
 
# # Training with X and y
# ppn.train(training_set_inputs, training_set_outputs)
ppn.train(X_train_flat, y_train)

# # # Plotting
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of iterations')
 
plt.tight_layout()
plt.show()

#Testing

# testing_set_inputs = np.array([[1,1,1,0,1,0,1],[1,0,1,0,1,0,1],[1,1,1,0,1,1,0],[1,0,1,0,1,1,0],
#                          [1,1,1,0,1,1,1],[1,0,1,0,1,1,1],[1,1,1,1,0,0,0],[1,0,1,1,0,0,0],
#                          [1,1,1,1,0,0,1],[1,0,1,1,0,0,1],[1,1,1,1,0,1,0],[1,0,1,1,0,1,0]])

# testing_set_outputs = np.array([[1,0,1,0,1,0,1,0,1,0,1,0]]).T


results = []


for x in (range(len(X_test_flat))):
    run = X_test_flat[x]
    trial = ppn.activation(run)
    results.append(trial.tolist())
    
    
print("RESULTS:")
            
print(tabulate({
                #"Input": X_test_flat,
                "Predicted value": results,
                "Actual value": y_test
               }, headers="keys"))    


