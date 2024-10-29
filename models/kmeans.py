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