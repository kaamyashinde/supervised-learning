import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate = 0.1, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
    
    def sigmoid_function(self, x):
        pass

    def _compute_loss(self, y, y_pred):
        pass

    def compute_gradients(self, x, y, y_pred):
        # b_0 + b_1 * x -> b_0 = grad_b, b_1 = grad_w
        feature_amount = len(x)
        datapoints_amount = len(x[0])
        collect_grad_b, collect_grad_w = [], []
        for f in feature_amount:
            grad_b = 0
            grad_w = 0
            for b in range(datapoints_amount):
                grad_b += (1 / datapoints_amount) * ((self.weights * x[f][b]) + self.bias - y[f][b])
            for w in x:
                grad_w += (1 / datapoints_amount) * (x[f][w]) * ((self.weights * x[f][w]) + self.bias - y[f][w])
            collect_grad_b.append(grad_b)
            collect_grad_w.append(grad_w)
        
        return collect_grad_b, collect_grad_w
        

    def update_parameters(self, grad_w, grad_b):
        pass

    def accuracy(true_values, predictions):
        return np.mean(true_values == predictions)
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        raise NotImplementedError("The fit method is not implemented yet.")
    
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
        # TODO: Implement
        raise NotImplementedError("The predict method is not implemented yet.")





