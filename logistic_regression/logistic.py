import numpy as np
from utils.sigmoid import sigmoid
from sklearn.datasets import load_iris
from utils.step_function import step_function


class Logistic_Regression():

    def __init__(self,  early_stop=True, iteration=10000, learning_rate=0.01, epsilon=0.00003):
        self.iter = iteration
        self.early = early_stop
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.paras = {}

    def _forward_propagate(self, X, w, b):
        z = np.dot(X, w) + b
        h = sigmoid(z)
        return h

    def _cost(self, h, y):
        m = y.shape[0]
        y1_cost = -np.log(h)
        y0_cost = -np.log(1 - h)
        return 1/m * np.sum(y * y1_cost + (1-y) * y0_cost)

    def fit(self, X, y):
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

            dz = (1/m) * (h - y)
            dw = np.dot(X.T, dz)
            assert(dw.shape == w.shape)
            db = np.sum(dz)

            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            # Stop if the step of gradient is small enough
            assert(prev_cost >= curr_cost)
            if (prev_cost - curr_cost) <= self.epsilon and self.early:
                break

        print("# of Iterations: ", i)
        self.paras["w"] = w
        self.paras["b"] = b

        # LR = model(name="Logistic Regression", paras=parameters)
        return self

    def __repr__(self):
        return "Logistic Regression:" + str(self.paras)

    def predict(self, X):
        h = self.predict_prob(X)
        print(h)
        for i in range(len(h)):
            h[i][0] = step_function(0.5, h[i][0])

        return h

    def predict_prob(self, X):
        w = self.paras["w"]
        b = self.paras["b"]

        h = self._forward_propagate(X, w, b)
        return h


if __name__ == "__main__":
    X = np.array([[0.7, 0.1, 0.9],
                  [0.1, 0.8, 0.1],
                  [0.8, 0.2, 0.8]])
    y = np.array([1, 0, 1]).reshape(3, 1)

    X_test = np.array([[0.99, 0.3, 0.99],
                      [0.8, 0.1, 0.7],
                      [0.3, 0.7, 0.3]])

    lr = Logistic_Regression(early_stop=False)
    p = lr.fit(X, y)
    print(p)

    y_hat = lr.predict(X_test)
    print(y_hat)
