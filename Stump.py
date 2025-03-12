import numpy as np

class Stump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_class = None    # prediction for feature < threshold
        self.right_class = None   # prediction for feature >= threshold

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        if sample_weight is None:
            # Initialize equal weights if not provided
            sample_weight = np.ones(n_samples) / n_samples
        else:
            # Normalize the sample weights
            sample_weight = sample_weight / np.sum(sample_weight)

        best_error = float('inf')
        # Iterate over all features and possible thresholds
        for feat in range(n_features):
            X_column = X[:, feat]
            # Consider thresholds between unique feature values
            unique_vals = np.sort(np.unique(X_column))
            # If only one unique value, use it as threshold
            # Otherwise use midpoints between consecutive values as candidates
            if len(unique_vals) == 1:
                thresholds = unique_vals  # degenerate case