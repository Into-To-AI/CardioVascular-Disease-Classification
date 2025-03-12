import numpy as np
from collections import Counter

class BaggingClassifier:
    def __init__(self, base_learner, n_estimators=10, sample_size=1.0):
        """
        Parameters:
        - base_learner: The base model (DecisionTree in this case)
        - n_estimators: Number of models in the ensemble
        - sample_size: Fraction of data to sample for each tree (default 1.0, full dataset size)
        1.0 means that the entire dataset is used to train each model
        0.5 means that half of the dataset is used to train each model
        0.3 means that 30% of the dataset is used to train each model
        1.5 means that 150% of the dataset is used to train each model with replacement
        """
        
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.models = []  # List to store trained models

    def _bootstrap_sample(self, X, y):
       
        n_samples = int(self.sample_size * len(X))
        indices = np.random.choice(len(X), n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """Train the bagging ensemble"""
        self.models = []  # Reset model list
        
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            model = self.base_learner() # Instantiate the base learner model of type DecisionTree
            model.fit(X_sample, y_sample)
            self.models.append(model)
        
    def predict(self, X):
        """Aggregate predictions from all base models using majority voting"""
        predictions = np.array([model.predict(X) for model in self.models])  # Shape: (n_estimators, n_samples)
        majority_votes = np.apply_along_axis(self._majority_vote, axis=0, arr=predictions) # axis = 0 means apply function on the rows [0, 1, 1, 0, 1]
        return majority_votes

    def _majority_vote(self, predictions):
        """Helper function to determine the majority class label"""
        counter = Counter(predictions) #Counts how many times each class appears. [0, 1, 1, 0, 1] -> {0: 2, 1: 3}
        return counter.most_common(1)[0][0] #Returns the class with the highest count