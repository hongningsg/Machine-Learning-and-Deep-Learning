import numpy as np
from utils import check_data
import pandas as pd
from sklearn.datasets import load_iris


class Naive_Bayes():

    def __init__(self, smooth=1):
        """Initialize NB classifier

           Parameters
           ----------
           distribution : str, {"Gaussian", "Multinomial", "Bernoulli"}
           Distribution used to calculate probability.

           Returns
           -------
           None
           """
        self.smooth = smooth

    def conditional_prob(self):
        pass


class MultinomialNB():

    def __init__(self, smooth=1):
        """Initialize NB classifier

           Parameters
           ----------
           smooth : int, default = 1 Leplace smoothing
                         if = 0 means no smoothing

           Returns
           -------
           None
           """
        self.smooth = smooth
        self.prior = None
        self.conditional = None

    def _prior_prob(self, y):
        y_unique, y_counts = np.unique(y, return_counts=True)
        self.prior = y_counts / len(y)

    def _split_data(self, X, y):
        X_splitted = {}
        full_data = np.concatenate((X, y), axis=1)
        y_unique = np.unique(y)
        for y_value in y_unique:
            data_temp = full_data[np.where(full_data[:,-1] == y_value)]
            X_temp = np.delete(data_temp, -1, axis=1)
            X_splitted[y_value] = X_temp
        return X_splitted

    def _conditional_prob(self, X_splitted):
        """Calculate the conditional probabilities P(X1=a|y=0), P(X1=b|y=0),...etc

           Parameters
           ----------
           X_splitted : dict, {0: (X with y = 0), 1: (X with y = 1), 2: (X with y = 2)...}

           Returns
           -------
           None
           """
        y_unique = list(X_splitted.keys()) # number of classes
        n = X_splitted[y_unique[0]].shape[1] # number of features
        counts = []
        totals = []
        for y in X_splitted.keys():
            temp_count = np.sum(X_splitted[y], axis=0).tolist()
            total = np.sum(X_splitted[y])
            counts.append(temp_count)
            totals.append(total)
        counts = np.array(counts)
        totals = np.array(totals).reshape(len(y_unique), 1)
        self.conditional = (counts + self.smooth) / (totals + n)
        # print(counts)
        # print(totals)

    def fit(self, X, y):
        """Fit NB classifier

           Parameters
           ----------
           X : array, (m, n)
           y : array, (m, 1)

           Returns
           -------
           None
           """
        X = check_data.check_X(X)
        y, c = check_data.check_y(y)
        self._prior_prob(y) # P(y=1), P(y=0), etc
        splitted_data = self._split_data(X, y) # dictionary with keys of y values
        # learn the conditional probability from data
        self._conditional_prob(splitted_data) # shape = (c, n)

    def predict(self, X):
        """Predict the data with self.prior and self.conditional

           Parameters
           ----------
           X : array, (m, n)
           y : array, (m, 1)

           Returns
           -------
           None
           """
        prior = self.prior
        cond_prob = self.conditional # shape=(c, n)
        P_class = np.ones(shape=(len(X), len(prior))) # shape=(m, c)
        for i in range(len(X)):
            prob = np.prod(np.power(cond_prob, X[i]), axis=1)
            prob = np.multiply(prior, prob)
            P_class[i] = prob
        prediction = np.argmax(P_class, axis=1)
        return prediction


if __name__ == "__main__":
    # iris = load_iris()
    # X, y = iris.data, iris.target

    nb = MultinomialNB()

    # Multinomial Toy Data
    # https: // www.youtube.com / watch?v = km2LoOpdB3A
    X = np.array([[2, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 1, 1]])

    y = np.array([1, 1, 1, 0])

    X_test = np.array([[3, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1]])
    y_test = np.array([[1], [0]])

    # run
    nb.fit(X, y)
    pred = nb.predict(X_test)
    print(nb.conditional)
    print(pred)

