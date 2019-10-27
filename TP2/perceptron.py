import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from tabulate import tabulate

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

class Perceptron:
    """Perceptron classifier."""
    def __init__(self, eta=0.1, n_iters=10):
        self.eta = eta
        self.n_iters = n_iters
 
    def train(self, X, y):
        """Function to train the neuron, which modifies the weights (w) based on the input values 
        and expected results.    
        """
        self.w_ = np.zeros(1 + X.shape[1])
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
        f = np.where(self.predict(X) >= 0.9, 1, 0)
    
        return f
 
    def predict(self, X):
        """Summation function."""
        # z = w · x + theta
        z = sigma(np.dot(X, self.w_[1:]) + self.w_[0])
        return z



        # Init the perceptron
ppn = Perceptron(eta=0.1, n_iters=10)
 
# Training with X and y
ppn.train(training_set_inputs, training_set_outputs)

# Plotting
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of iterations')
plt.tight_layout()
plt.show()

# Testing

testing_set_inputs = np.array([[1,1,1,0,1,0,1],[1,0,1,0,1,0,1],[1,1,1,0,1,1,0],[1,0,1,0,1,1,0],
                         [1,1,1,0,1,1,1],[1,0,1,0,1,1,1],[1,1,1,1,0,0,0],[1,0,1,1,0,0,0],
                         [1,1,1,1,0,0,1],[1,0,1,1,0,0,1],[1,1,1,1,0,1,0],[1,0,1,1,0,1,0]])

testing_set_outputs = np.array([[1,0,1,0,1,0,1,0,1,0,1,0]]).T


results = []

for x in (range(len(testing_set_inputs))):
    run = testing_set_inputs[x]
    trial = ppn.activation(run)
    results.append(trial.tolist())
    
    
print("RESULTS:")
            
print(tabulate({
                "Input": testing_set_inputs,
                "Predicted value": results,
                "Actual value": testing_set_outputs
               }, headers="keys"))    