import numpy as np
from .base_model import BaseModel

class CustomKNN(BaseModel):
    """Custom implementation of K-Nearest Neighbors"""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Calculate distances
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
            
        return np.array(predictions)
