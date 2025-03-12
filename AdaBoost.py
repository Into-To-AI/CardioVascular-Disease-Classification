import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, number_of_stumps=50):
        self.number_of_weak_learners = number_of_stumps
        self.stumps = []
        self.stump_weights = []
        self.classes = None

    def fit(self, X_train, Y_train):

        # Convert class labels to +1/-1 (sign calculation)
        y_mapped = np.where(Y_train <= 0, -1, 1)
        self.classes = np.unique(Y_train)


        # Initialize sample weights
        number_of_samples = X_train.shape[0]
        #initially all having the same weight (1/#samples)
        weights = np.ones(number_of_samples) / number_of_samples  


        for t in range(self.number_of_weak_learners):

            # Train a weighted decision stump (max depth = 1)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X_train, y_mapped, sample_weight=weights)
            stump_pred = stump.predict(X_train)

            # Compute weighted error
            misclassified = stump_pred != y_mapped
            #misclssified is a boolean array representing the classification incorrectness.
            error = np.dot(weights, misclassified.astype(float)) / np.sum(weights)

            # Encounter perfect / very bad weak learner, stop training
            if error == 0:
                alpha = 1e9  # Perfect classifier, give it very high weight
                self.stumps.append(stump)
                self.stump_weights.append(alpha)
                break
            # worse than chance
            if error >= 0.5:
                continue  # Stop if weak learner is as bad as random guessing

            # Compute weak learner weight (alpha)
            alpha = 0.5 * np.log((1.0 - error) / error)

            # Update sample weights
            weights = weights * np.exp(-alpha * y_mapped * stump_pred)
            # Normalize sample weights
            weights /= np.sum(weights)  

            # Append Learner
            self.stumps.append(stump)
            self.stump_weights.append(alpha)

    def predict(self, X):
        
        num_samples = X.shape[0]
        agg = np.zeros(num_samples)
        for alpha, stump in zip(self.stump_weights, self.stumps):
            agg += alpha * stump.predict(X)

        # Convert back to original labels
        y_pred_mapped = np.sign(agg)
        y_pred = np.where(y_pred_mapped <= 0, 0, 1)
        return y_pred
