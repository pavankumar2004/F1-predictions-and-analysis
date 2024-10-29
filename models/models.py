# models.py
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

class CustomLinearRegression(BaseModel):
    """Improved implementation of Linear Regression using Gradient Descent with feature scaling"""
    
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


class CustomLogisticRegression(BaseModel):
    """Custom implementation of Logistic Regression using Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

import numpy as np

class CustomKMeans:
    """Custom implementation of K-Means clustering"""
    
    def __init__(self, n_clusters=8, max_iters=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        
    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign samples to closest centroids
            old_centroids = self.centroids.copy()
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:
                    self.centroids[k] = np.mean(X[self.labels_ == k], axis=0)
            
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break
                
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def fit_predict(self, X):
        """Fit the model and return the labels."""
        self.fit(X)
        return self.labels_

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

class CustomStandardScaler:
    """Custom implementation of StandardScaler"""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class CustomLabelEncoder:
    """Custom implementation of LabelEncoder"""
    
    def __init__(self):
        self.classes_ = None
        self.class_dict = None
    
    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_dict = {c: i for i, c in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        return np.array([self.class_dict[yi] for yi in y])
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        inverse_dict = {i: c for c, i in self.class_dict.items()}
        return np.array([inverse_dict[yi] for yi in y])

class CustomDecisionTree:
    """A simple decision tree classifier from scratch."""

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    class Node:
        """Node class for the decision tree."""
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        """Fit the model to the training data."""
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Stopping conditions
        if len(unique_classes) == 1:
            return self.Node(value=unique_classes[0])
        if self.max_depth is not None and depth >= self.max_depth:
            most_common_label = np.bincount(y).argmax()
            return self.Node(value=most_common_label)

        # Best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            most_common_label = np.bincount(y).argmax()
            return self.Node(value=most_common_label)

        # Split data
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return self.Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, X, y):
        """Find the best split for the data."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, feature_column, threshold):
        """Calculate the information gain of a split."""
        parent_entropy = self._entropy(y)
        left_indices = feature_column < threshold
        right_indices = feature_column >= threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])

        # Weighted average of the child entropies
        child_entropy = (n_left / n) * self._entropy(y[left_indices]) + (n_right / n) * self._entropy(y[right_indices])

        return parent_entropy - child_entropy

    def _entropy(self, y):
        """Calculate the entropy of a label array."""
        hist = np.bincount(y)
        probs = hist / len(y)
        return -np.sum(p * np.log2(p + 1e-9) for p in probs if p > 0)

    def predict(self, X):
        """Make predictions for the input data."""
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, node):
        """Predict a single sample using the tree."""
        if node.value is not None:
            return node.value

        if sample[node.feature] < node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
