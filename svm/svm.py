import numpy as np
from scipy.optimize import fsolve


class svm():

    def __init__(self, margin='hard', solver='dual', kernel='linear'):
        # solver = 'dual' or 'smo'
        self.w = None # (n,1)
        self.b = 0
        self.margin = margin
        self.kernel = kernel
        self.solver = solver

    def _linear_kernel(self, X, y, b=1):
        """
        K(x,z) = x.T * z + b
        
        :param X: 
        :param y: 
        :return: 
        """
        return X.T @ y + b

    def _poly_kernel(self, X, y, b=1, degree=2):
        """
        Returns the gaussian similarity of arrays `x` and `y` with
        kernel width parameter `sigma` (set to 1 by default).
        :param X: 
        :param y: 
        :param sigma: 
        :return: 
        """
        return (X.T @ y + b) ** degree

    def split_data(self, X, y):
        pos = []
        neg = []
        for i, v in enumerate(y):
            if v == 0:
                neg.append(X[i])
            else:
                pos.append(X[i])
        data_dict = {k:v for k in (-1, 1) for v in (np.array(neg), np.array(pos))}
        return data_dict

    def fit(self, X, y):
        if self.solver == 'dual':
            for i in range(len(X)):
                pass

    def _gram_matrix(self, X, y):
        """
        
        :param X: (m, n)
        :param y: (m, 1)
        :return: 
        """
        X_prime = np.multiply(X, y) #(m,n)
        return np.dot(X_prime, X_prime.T) #(m,m)

    def predict(self):
        pass
