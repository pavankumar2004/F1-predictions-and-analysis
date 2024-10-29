import numpy as np
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
