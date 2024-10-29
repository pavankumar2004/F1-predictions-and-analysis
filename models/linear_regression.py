import numpy as np
from .base_model import BaseModel

class CustomLinearRegression(BaseModel):
    """implementation of Linear Regression using Gradient Descent with feature scaling"""
    
    def __init__(self, learning_rate=0.001, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def _scale_features(self, X):
        """Standardize features by removing the mean and scaling to unit variance"""
        if self.scaler_mean is None:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0)
        # Avoid division by zero
        self.scaler_std = np.where(self.scaler_std == 0, 1, self.scaler_std)
        return (X - self.scaler_mean) / self.scaler_std
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Scale features
        X_scaled = self._scale_features(X)
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Track loss for convergence checking
        prev_loss = float('inf')
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X_scaled, self.weights) + self.bias
            
            # Compute loss
            current_loss = np.mean((y_predicted - y) ** 2)
            
            # Check for convergence
            if abs(prev_loss - current_loss) < 1e-7:
                break
            prev_loss = current_loss
            
            # Compute gradients with normalization
            dw = (1/n_samples) * np.dot(X_scaled.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters with gradient clipping
            dw = np.clip(dw, -1, 1)
            db = np.clip(db, -1, 1)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self
    
    def predict(self, X):
        # Scale features using training statistics
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        return np.dot(X_scaled, self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        return 1 - (ss_residual / ss_total)