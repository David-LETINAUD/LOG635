import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from tabulate import tabulate
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize

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
