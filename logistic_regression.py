import numpy as np
import math 

class LogisticRegression():
    
    def __init__(self, learning_rate = 0.1, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0
        self.losses, self.train_accuracies = [], []
    
    def sigmoid_function(self, x):
        """
        Scales the values to the interval [0, 1]
        """
        return 1 / (1 + math.exp((-self.weights @ x) + self.bias ))

    def _compute_loss(self, y, y_pred):
        """
        Uses cross-entropy loss
        """
        return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def compute_gradients(self, X, y, y_pred):
        """
        First looks at how many features we have. Then for each feature, it iterates and finds the weights for the different features. Then these weights are added to a list for grad_w and is returned. The bias on the other hand is only one, considering it is a linear regression. 
        """
        m = X.shape[0] #number of samples

        # Gradients
        error = y_pred - y
        grad_w = (X.T @ error) / m 
        grad_b = np.mean(error)

        return grad_w, grad_b
        

    def update_parameters(self, grad_w, grad_b):
        """
        Gets the weights and bias arrays saved in the class.
        Iterates over each value in the arrays and updates it with the help of the learning rate and the gradient found.
        """
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def accuracy(self, true_values, predictions):
        pass

    def normalise(self, X):
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)

        return ((X - self.mean) / self.std)
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        X = self.normalise(X)
        
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias
            
            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)

            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]

            accuracies = self.accuracy(y, pred_to_class)
            self.train_accuracies.append(accuracies)
            self.losses.append(loss)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        X = self.normalise(X)

        return np.matmul(X, self.weights) + self.bias
        





