import sklearn.decomposition as dp
import numpy as np
from utils import normalize

class pca():

    def __init__(self, dim):
        self.dim = dim
        pass

    def svd(self, X):
        return np.linalg.svd(X, full_matrices=False)

    def pca(self, X):
        mean = np.mean(X, axis=0)
        X -= mean
        # returned s already been sorted, from max to min
        u, s, vh = self.svd(X)
        s = np.diag(s)[:, :self.dim]
        print(u.shape)
        print(s.shape)
        return np.dot(u,s)

if __name__ == '__main__':
    X = np.array([[1.2, 2, 3],
                  [20, 5, 6],
                  [12, 4, 9]])
    X = np.random.rand(20,10)
    pca = pca(3)
    T = pca.pca(X)
    print(T)

    p = dp.PCA(n_components=3)
    X = p.fit_transform(X)
    print(X)
