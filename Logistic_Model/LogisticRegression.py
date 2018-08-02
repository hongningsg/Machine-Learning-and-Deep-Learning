import numpy as np

__all__ = ["LogisticRegression"]

class LogisticRegression():
    def __init__(self, max_iter = 100, learn_rate = 0.1):
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.model = None

    def __LogisticFunction(self, x):
        return 1/(1+np.exp(-x))

    def __dG_dw(self,w):
        return self.LogisticFunction(w)*(1 - self.LogisticFunction(w))

    def __gradientDescent(self, W, X, Y):
        m, n = X.shape
        num_iter = 0
        while num_iter < self.max_iter:
            num_iter += 1
            print("W X", W.shape, X.shape)
            Z = np.dot(X, W)
            print('Z',Z.shape)
            G = self.__LogisticFunction(Z)
            print('G',G.shape)
            D = G - Y
            print("D", D.shape)
            print("X", X.shape)
            dW = np.dot(X.T, D)
            # dW = (np.dot(np.transpose(G - Y),X)) / X.shape[0]
            print('DW',dW.shape)
            W = W - self.learn_rate * dW
        return W


    def fit(self, X, Y):
        W = np.zeros((X.shape[1], 1))
        W = self.__gradientDescent(W, X, Y)
        self.model = W
        return self.model

