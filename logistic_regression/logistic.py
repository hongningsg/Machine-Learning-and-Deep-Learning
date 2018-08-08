import numpy as np
import time
from copy import deepcopy
from utils.sigmoid import sigmoid, step_function
from utils.check_data import check_data
from sklearn.datasets import load_iris

# Author: Zhengting Li <heyimzhengting@gmail.com>

class Logistic_Regression():

    def __init__(self,  early_stop=False, iteration=10000, learning_rate=0.01, epsilon=0.00003):
        self.iter = iteration
        self.early = early_stop
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.paras = {}

    def __repr__(self):
        return "Logistic Regression:" + str(self.paras)

    def _forward_propagate(self, X, w, b):
        """Computes h = sigmoid(w*x + b)

           Parameters
           ----------
           X : ndarray, shape (m_samples, n_features)
               Training data.
           w : ndarray, shape (n_features, 1)
               Coefficient vector.
           y : ndarray, shape (m_samples, 1)
               Array of labels.
               
           Returns
           -------
           h : ndarray, shape(m_samples, 1)
               Activation values (hypothesis) (probability to be class 1)
           """
        z = np.dot(X, w) + b
        h = sigmoid(z)
        return h

    def _cost(self, h, y):
        m = y.shape[0]
        y1_cost = -np.log(h)
        y0_cost = -np.log(1 - h)
        cost = 1/m * np.sum(y * y1_cost + (1-y) * y0_cost)
        return cost

    def _train(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        w, b = np.zeros((n, 1)), 0

        i = 0
        costs = []
        curr_cost = np.inf
        while i < self.iter:
            i += 1

            h = self._forward_propagate(X, w, b)
            # print(h)
            prev_cost = curr_cost
            curr_cost = self._cost(h, y)
            costs.append(curr_cost)

            dz = (1 / m) * (h - y)
            dw = np.dot(X.T, dz)
            assert (dw.shape == w.shape)
            db = np.sum(dz)

            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            # Stop if the step of gradient is small enough
            assert (prev_cost >= curr_cost)
            if (prev_cost - curr_cost) <= self.epsilon and self.early:
                break

        paras = {}
        paras["w"] = w
        paras["b"] = b
        return paras

    def fit(self, X, y):
        start = time.time()
        X, y, c = check_data(X, y) # nb of classes

        if c == 2:
            self.paras = self._train(X, y)
        elif c > 2:
            for i in range(c):
                y_copy = deepcopy(y)
                for j in range(len(y)):
                    if y_copy[j][0] == i:
                        y_copy[j][0] = 1
                    else:
                        y_copy[j][0] = 0
                self.paras[i] = self._train(X, y_copy)

        stop = time.time()
        print("Time taken: ", "{0:.3}".format(stop - start), "seconds")
        return self.paras

    def fit_plot(self):
        pass

    def predict(self, X):
        c = len(self.paras.keys()) # number of classes (1 means bi-class, 3+ means multi-class)
        h = self.predict_prob(X)
        if c > 2:
            h = np.argmax(h, axis=1)
        else:
            for i in range(len(h)):
                h[i][0] = step_function(0.5, h[i][0])

        return h

    def predict_prob(self, X):
        c = len(self.paras.keys())  # number of classes (1 means bi-class, 3+ means multi-class)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if c > 2: # multi-class
            m = X.shape[0]
            H = []
            for i in range(c):
                parameter = self.paras[i]
                w = parameter["w"]
                b = parameter["b"]
                h_i = self._forward_propagate(X, w, b)
                H.append(h_i)
        else: # bi-class
            w = self.paras["w"]
            b = self.paras["b"]
            H = self._forward_propagate(X, w, b)
        return np.array(H).reshape(c, m).T # reshape it to (m, c)


if __name__ == "__main__":
    # X = np.array([[0.7, 0.1, 0.9],
    #               [0.1, 0.8, 0.1],
    #               [0.8, 0.2, 0.8]])
    # y = np.array([1, 0, 1]).reshape(3, 1)
    #
    # X_test = np.array([[0.99, 0.3, 0.99],
    #                   [0.8, 0.1, 0.7],
    #                   [0.3, 0.7, 0.3]])
    #

    iris = load_iris()
    X, y = iris.data, iris.target

    lr = Logistic_Regression()
    p = lr.fit(X, y)

    h = lr.predict_prob(X)
    y_hat = lr.predict(X)
    print(y_hat)
    print(iris.target)
