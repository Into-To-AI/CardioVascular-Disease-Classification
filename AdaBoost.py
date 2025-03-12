import numpy as np
from collections import Counter
from Stump import Stump

class AdaBoost:
    def __init__(self, number_of_learners=50):
        self.number_of_learners = number_of_learners
        
    def fit(self, x_train, y_train):
        # Convert labels to -1/+1
        y = np.where(y_train <= 0, -1, 1)
        
        number_of_samples = x_train.shape[0]
        self.weights = np.full(number_of_samples, fill_value=(1 / number_of_samples))

        self.weak_learners = []
        self.weak_learner_weights = []

        for i in range(self.number_of_learners):
            # Create new stump for each iteration
            h_t = Stump()
            h_t.fit(x_train, y, self.weights)
            self.weak_learners.append(h_t)

            # Calculate error
            predictions = h_t.predict(x_train)
            incorrect = predictions != y
            error = np.sum(self.weights[incorrect]) / np.sum(self.weights)

            # Handle edge cases
            error = np.clip(error, 1e-15, 1 - 1e-15)
            
            # Calculate learner weight
            alpha_t = 0.5 * np.log((1 - error) / error)
            self.weak_learner_weights.append(alpha_t)

            # Update sample weights
            self.weights *= np.exp(-alpha_t * y * predictions)
            self.weights /= np.sum(self.weights)  # Normalize
    
    def predict(self, x_test):
        #initialize the predictions
        predictions = np.zeros(x_test.shape[0])
        
        #for each base learner, make a prediction and multiply it by the weight of the base learner
        for i, h_t in enumerate(self.weak_learners):
            predictions += self.weak_learner_weights[i] * h_t.predict(x_test)
        
        #return the sign of the predictions
        return np.sign(predictions)





