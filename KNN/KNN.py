import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        precdications = [self._predict(x) for x in X]
        return precdications

    def _predict(self, X):
        #calculate dsitance
        distances = [euclidean_distance(X, x_train) for x_train in self.X_train]

        #get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]