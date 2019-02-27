import numpy as np
from sklearn.datasets import load_iris

class KMeans():

    def __init__(self, k, init ='random', n_init=10, max_iter=300):
        self.k = k
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter

    def _init_centroids(self, X):
        m = X.shape[0]
        if self.init == 'k_means++':
            # initiate with 1 random centroid
            centroids = [X[np.random.choice(m, 1)]]
            n_centroids = 1
            # initialize distance matrix with one distance
            # (n_centroids, n)
            D = np.empty(shape=(m, self.k))
            while n_centroids < self.k:
                for i in range(len(X)):
                    # only update one col of distance
                    D[i][n_centroids-1] = self._distance(X[i], centroids[n_centroids-1])
                # (...) make sure same dimension
                D_current = D[..., :n_centroids]
                # print(D_current.shape)
                D_min = np.min(D_current, axis=1)
                D_sum = sum(D_min ** 2)
                prob = D_min ** 2 / D_sum
                c = X[np.random.choice(m, 1, p=prob)]
                centroids.append(c)
                n_centroids += 1
            return centroids

        elif self.init == 'random':
            return X[np.random.choice(m, self.k, replace=False)]
        else:
            pass

    def _distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def fit(self, X):
        m = X.shape[0]
        init_centroids = self._init_centroids(X)
        # centroids dictionary
        centroids = {k:v for k,v in enumerate(init_centroids)}
        last_centroids = {k:[] for k,v in enumerate(init_centroids)}
        iteration = 0
        while iteration < self.max_iter:
            for k in centroids: # stop if the centroids doesn't change between two iterations
                if np.array_equal(centroids[k], last_centroids[k]):
                    break
            last_centroids = centroids.copy()
            # init dictionary to store points and distance matrix
            pts = {k: [] for k in centroids.keys()}
            D = np.empty(shape=(m, len(centroids)))
            for i in range(m):
                # j is the clustering, c is the centroid vector
                for j,c in centroids.items():
                    D[i][j] = self._distance(X[i], c)
            arg_min = np.argmin(D, axis=1)
            assert(len(arg_min) == m)
            for i, min in enumerate(arg_min):
                pts[min].append(X[i])
            pts = {k:np.array(matrix) for k, matrix in pts.items()}
            for k,matrix in pts.items():
                centroids[k] = np.mean(matrix, axis=0)
            iteration += 1
        return centroids, pts

if __name__ == "__main__":
    model = KMeans(k=5, init='k_means++')
    data = load_iris()
    X, y = data.data, data.target
    points = model._init_centroids(X)
    centroids, pts = model.fit(X)
    print(centroids)
    for i,v in pts.items():
        print(len(v))
    print(len(X))
