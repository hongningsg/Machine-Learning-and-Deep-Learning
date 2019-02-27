import numpy as np
from utils import check_data


class KNNBase():
    def __init__(self, k, norm=2):
        self.k = k
        self.norm = norm
        self.X = None
        self.y = None
        # assert(len(self.X) == len(self.y))

    def _distance(self, xi, xj):
        if not xi.shape == xj.shape:
            raise ValueError('xi and xj must be same shape in order to compute the distance')
        # Minkowski distance
        D = np.power(np.sum((abs(xi - xj))**self.norm), 1/self.norm)
        return D


class KNNClassifier(KNNBase):

    def __init__(self, k, norm=2):
        super().__init__(k, norm=norm)

    def fit(self, X, y):
        self.X = check_data.check_X(X)
        self.y = y

    def predict(self, X_test):
        m = len(self.X)
        m_test = len(X_test)
        self.distance_matrix = np.zeros((m_test, m))
        for i in range(m_test):
            for j in range(m):
                self.distance_matrix[i][j] = self._distance(X_test[i], self.X[j])
        index = np.argsort(self.distance_matrix, axis=1) # return the indexes of sorted rows
        index = np.flip(index, axis=1) # change it from max to min
        print(index)
        # pick k indexes and find the corresponding y values
        y_unique = np.unique(self.y)
        assert(len(index) == X_test.shape[0])
        prediction = np.empty(shape=(m, 1))
        # record the counts of the class with k nearest neighbours in a dict
        for i in range(len(index)):
            y_counts = {k: 0 for k in y_unique}
            for j in range(self.k):
                id = index[i][j]
                y_counts[self.y[id]] += 1
            print(y_counts)
            pred = max(y_counts, key=y_counts.get) # return the class with the max counts in the dict
            print(pred)
            print(prediction.shape)
            prediction[i][0] = pred
        return prediction



class KNNRegression(KNNBase):

    def __init__(self, k):
        super().__init__(k, norm=2)


