import numpy as np
from collections import Counter
from Node import Node

class DecisionTree:
    # n_feats is the number of features to consider when looking for the best split
    def __init__(self, min_sample_split=10, max_depth=5, n_feats=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        n_features = X.shape[1]
    
        # Calculate lower bound (half the features)
        lower_bound= n_features // 2
        rng = np.random.RandomState(42)
        # Randomly choose the number of features to select, between lower_bound and n_features.
        self.n_feats = rng.randint(lower_bound, n_features + 1)
        self.n_feats = X.shape[1] if self.n_feats is None else min(X.shape[1], self.n_feats)
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samp, n_feat = X.shape
        n_labs = len(np.unique(y))

        if depth >= self.max_depth or n_samp < self.min_sample_split or n_labs == 1:
            val = self._majority(y)
            return Node(value=val)

        rng = np.random.RandomState(42)
        feature_idxs = rng.choice(n_feat, self.n_feats, replace=False)

        best_thresh, best_feat_idx = self._best_split(X, y, feature_idxs)
        
        left_idxs, right_idxs = self._split_node(X[:, best_feat_idx], best_thresh)

        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feat_idx, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = float('-inf')
        best_feat = None
        best_thres = None

        for feature in feat_idxs:
            X_col = X[:, feature]
            thresholds = np.unique(X_col)

            for thres in thresholds:
                gain = self._calculate_info_gain(X_col, y, feature, thres)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thres = thres

        return best_thres, best_feat

    def _split_node(self, X_col, threshold):
        left = np.argwhere(X_col <= threshold).flatten()
        right = np.argwhere(X_col > threshold).flatten()
        return left, right

    def _calculate_info_gain(self, X_col, y, feature, threshold):
        root_entropy = self._entropy(y) 

        left, right = self._split_node(X_col, threshold)

        n_left, n_right = len(left), len(right)
        left_entropy = self._entropy(y[left])
        right_entropy = self._entropy(y[right])
        children_entropy = (n_left / len(y)) * left_entropy + (n_right / len(y)) * right_entropy

        return root_entropy - children_entropy

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0]) 
    
    def _majority(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node is None:
            return None
        if node.left is None and node.right is None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
