import numpy as np
from Node import Node

class Stump:
    def __init__(self):
        self.root = None

    def fit(self, X, y, weights):
        self.n_feats = X.shape[1]
        self.root = self._build_stump(X, y, weights)

    def _build_stump(self, X, y, weights):
        n_samples, n_features = X.shape
        
        best_thresh, best_feat_idx = self._best_split(X, y, weights)
        
        # Split the data
        left_mask = X[:, best_feat_idx] <= best_thresh
        right_mask = ~left_mask
        
        # Get weighted majority class for each split
        left_value = self._weighted_majority(y[left_mask], weights[left_mask])
        right_value = self._weighted_majority(y[right_mask], weights[right_mask])
        
        left = Node(value=left_value)
        right = Node(value=right_value)
        
        return Node(feature=best_feat_idx, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, weights):
        min_error = float('inf')
        best_feat = 0
        best_thresh = 0

        for feature in range(self.n_feats):
            X_col = X[:, feature]
            thresholds = np.unique(X_col)

            for thresh in thresholds:
                left_mask = X_col <= thresh
                right_mask = ~left_mask
                
                # Skip if split creates empty node
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                    
                error = self._weighted_error(y, weights, left_mask, right_mask)
                
                if error < min_error:
                    min_error = error
                    best_feat = feature
                    best_thresh = thresh

        return best_thresh, best_feat

    def _weighted_majority(self, y, weights):
        if len(y) == 0:
            return 0
        unique_classes = np.unique(y)
        class_weights = [np.sum(weights[y == c]) for c in unique_classes]
        return unique_classes[np.argmax(class_weights)]

    def _weighted_error(self, y, weights, left_mask, right_mask):
        left_pred = self._weighted_majority(y[left_mask], weights[left_mask])
        right_pred = self._weighted_majority(y[right_mask], weights[right_mask])
        
        left_error = np.sum(weights[left_mask] * (y[left_mask] != left_pred))
        right_error = np.sum(weights[right_mask] * (y[right_mask] != right_pred))
        
        return (left_error + right_error) / np.sum(weights)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        return self._traverse(x, node.left) if x[node.feature] <= node.threshold else self._traverse(x, node.right)



