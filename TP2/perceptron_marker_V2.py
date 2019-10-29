import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

LABELS = [ "Circle2", "Circle5", "Diamond2","Diamond5", "Hexagon2", "Hexagon5", "Triangle2", "Triangle5"]


#################################  Functions
def sigma(x):
    return 1 / (1 + np.exp(-x))

def flatten_x(x):
    return np.array([i.flatten() for i in x])

def vectorize_y(y):
    y_vec = []
    for str_lbl in y :
        y_vec.append( [1*(i==str_lbl) for i in LABELS] )
    return np.array(y_vec)

def accuracy(y_predict,y):
    y_pred_int = np.array( [np.argmax(p) for p in y_predict])
    y_int = np.array( [np.argmax(p) for p in y])
    VP = np.sum( np.equal(y_pred_int, y_int) )
    return VP/len(y_int)

#################################  Perceptron Class
class Perceptron:
    """Perceptron classifier."""
    def __init__(self, eta=0.1, n_iters=10):
        self.eta = eta
        self.n_iters = n_iters
 
    def train(self, X, y):
        """Function to train the neuron, which modifies the weights (w) based on the input values 
        and expected results.    
        """
        self.w_ = np.zeros( (8,1 + X.shape[1]) )        
        self.errors_ = []

        for _ in range(self.n_iters):
            errors =  np.zeros(8)
            for xi, target in zip(X, y):
                error = target - self.activation(xi)
                update = self.eta * error
                cpt = 0
                for i in self.w_:
                    i[1:] += update[cpt] * xi
                    i[0] += update[cpt]
                    cpt+=1
                    
                #errors += int(update != 0.0)
                errors += [int(i != 0.0) for i in update ]
            self.errors_.append(errors)
        return self
 
    def activation(self, X):
        """Return class using Heaviside step function
        f(z) = 1 if z >= theta; 0 otherwise.
        """
        f = np.zeros(8)
        #f = np.where(self.predict(X) >= 0.9, 1, 0)
        d = self.predict(X)
        f[np.argmax(d)] = 1
        #f = np.around(self.predict(X), decimals=2)
        return f
 
    def predict(self, X):
        """Summation function."""
        #z = w · x + theta
        z = np.zeros(8)
        cpt=0
        for i in self.w_:
            tmp = np.dot(X, i[1:]) + i[0]
            z[cpt] = sigma(tmp)
            cpt+=1
            
        #z = softmax(np.dot(X, self.w_[1:]) + self.w_[0])
        return z

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

print(data_size, X_train.shape, y_train.shape)
print(data_size, X_test.shape, y_test.shape)

# Init the perceptron
ppn = Perceptron(eta=0.1, n_iters=50)
 
# # Training with X and y
ppn.train(X_train, y_train)

# # # Plotting
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of iterations')
 
plt.tight_layout()
plt.show()


# Testing
results = []

for x in (range(len(X_test))):
    run = X_test[x]
    trial = ppn.activation(run)
    results.append(trial.tolist())
    
print("RESULTS:")
            
print(tabulate({
                #"Input": X_test_flat,
                "Predicted value": results,
                "Actual value": y_test
               }, headers="keys"))    

acc = accuracy(results,y_test)
print(acc)