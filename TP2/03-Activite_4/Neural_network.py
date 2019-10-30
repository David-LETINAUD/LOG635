import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_prime(z):
    o = sigmoid(z)  # Pour ne calculer qu'1 fois sigmoid(z)
    return o*(1-o)

def cross_entropy(outputs, y_target):
    return -np.sum(y_target*np.log(outputs))

def summation(w, X):
    return np.dot(w,X)

def one_hot(y, n_labels):
    mat = np.zeros((len(y), n_labels))
    for i, val in enumerate(y):
        mat[i, val] = 1
    return mat

def mle(y, axis=1):
    return np.argmax(y, axis)

def softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

class NNClassifier:

    def __init__(self, n_classes, n_features, n_hidden_units=30, epochs=500, learning_rate=0.01,
                 n_batches=1):
        # Parameters of the network
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.n_batches = n_batches
        self.w1, self.w2 = self.weights()

        self.epochs = epochs
        self.learning_rate = learning_rate

    def weights(self):
        # Weight matrix from input to hidden layer
        w1 = np.random.uniform(-1.0, 1.0,
                               size=(self.n_hidden_units, self.n_features))

        # Weight matrix from hidden to output layer
        w2 = np.random.uniform(-1.0, 1.0,
                               size=(self.n_classes, self.n_hidden_units))

        return w1, w2

    def forward(self, X):
        # dot product of X (input) and first set of weights
        z1 = summation(self.w1, X.T)  # self.w1.dot(X.T)
        hidden_output = sigmoid(z1)  # activation function

        # dot product of hidden layer and second set of weights
        z2 = summation(self.w2, hidden_output)  # self.w2.dot(hidden_output)
        final_output = sigmoid(z2)   # final activation function

        return z1, hidden_output, z2, final_output
    
    def error(self, y, output):
        error = cross_entropy(output, y)
        return 0.5 * np.mean(error)

    def backward(self, X, z1, hidden_output, final_output, y):
        
        # Error in output
        output_error = final_output - y

        # Applying derivative of sigmoid to error
        output_delta = self.w2.T.dot(output_error) * sigmoid_prime(z1)

        # How much our input layer weights contributed to output error
        grad1 = output_delta.dot(X)

        # How much our hidden layer weights contributed to output error
        grad2 = output_error.dot(hidden_output.T)
        
        return grad1, grad2   

    def backprop_step(self, X, y):
        z1, hidden_output, z2, final_output = self.forward(X)
        y = y.T

        grad1, grad2 = self.backward(X, z1, hidden_output, final_output, y)
        
        # Calculating the error with the cross-entropy function
        error = self.error(y, final_output)

        return error, grad1, grad2   

    def train(self, X, y):
        self.error_ = []        
        y = one_hot(y, self.n_classes)
        
        # Divides the dataset in processing bathces
        X_bathces = np.array_split(X, self.n_batches)
        y_bathces = np.array_split(y, self.n_batches)      

        # Make the iterations
        for i in range(self.epochs):

            epoch_errors = []

            for Xi, yi in zip(X_bathces, y_bathces):

                # Update weights
                error, grad1, grad2 = self.backprop_step(Xi, yi)
                epoch_errors.append(error)
                self.w1 -= (self.learning_rate * grad1)
                self.w2 -= (self.learning_rate * grad2)
            self.error_.append(np.mean(epoch_errors))
        return self
    
    def predict(self, X):        
        z1, hidden_output, z2, final_output = self.forward(X)
        return mle(z2.T)

    def predict_proba(self, X):        
        z1, hidden_output, z2, final_output = self.forward(X)
        return softmax(final_output.T)

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.sum(y == y_hat, axis=0) / float(X.shape[0])