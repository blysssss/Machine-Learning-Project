"""
Logistic Regression Implementation from Scratch
A binary classification algorithm using sigmoid activation and gradient descent.
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent optimization.
    n_iterations : int, default=1000
        Number of iterations for gradient descent.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _sigmoid(self, z):
        """
        Compute sigmoid activation function.
        
        Parameters:
        -----------
        z : array-like
            Linear combination of weights and features.
            
        Returns:
        --------
        array-like
            Sigmoid activation values.
        """
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Parameters:
        -----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted probabilities.
            
        Returns:
        --------
        float
            Binary cross-entropy loss.
        """
        m = len(y_true)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values (0 or 1).
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward propagation
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward propagation
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability estimates.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted probabilities.
        """
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels.
            
        Returns:
        --------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
