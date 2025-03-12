import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.stump_weights = []
        self.classes = None
        self.class_mapping = None

    def fit(self, X, y):
        # Convert class labels to +1/-1
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("AdaBoost implementation only supports binary classification.")

        mapping = {self.classes[0]: -1, self.classes[1]: 1}
        y_mapped = np.array([mapping[label] for label in y])
        self.class_mapping = mapping

        # Initialize sample weights
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # weight = 1/N

        self.stumps = []
        self.stump_weights = []

        for t in range(self.n_estimators):
            # Train a weighted decision stump (max depth = 1)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y_mapped, sample_weight=w)
            stump_pred = stump.predict(X)

            # Compute weighted error
            misclassified = stump_pred != y_mapped
            error = np.dot(w, misclassified.astype(float)) / np.sum(w)

            # Stop if error is too high (random guess) or perfect classifier
            if error <= 0:
                alpha = 1e9  # Perfect classifier, give it very high weight
                self.stumps.append(stump)
                self.stump_weights.append(alpha)
                break
            if error >= 0.5:
                break  # Stop if weak learner is as bad as random guessing

            # Compute weak learner weight (alpha)
            alpha = 0.5 * np.log((1.0 - error) / error)
            self.stumps.append(stump)
            self.stump_weights.append(alpha)

            # Update sample weights
            w = w * np.exp(-alpha * y_mapped * stump_pred)
            w /= np.sum(w)  # Normalize weights

    def predict(self, X):
        if not self.stumps:
            raise Exception("No stumps have been trained.")

        # Aggregate weak learner predictions weighted by alpha
        agg = np.zeros(X.shape[0])
        for alpha, stump in zip(self.stump_weights, self.stumps):
            agg += alpha * stump.predict(X)

        # Convert back to original labels
        y_pred_mapped = np.sign(agg)
        inv_map = {v: k for k, v in self.class_mapping.items()}
        y_pred = np.array([inv_map[val] for val in y_pred_mapped])
        return y_pred
