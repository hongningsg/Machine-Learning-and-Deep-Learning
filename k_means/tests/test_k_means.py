from sklearn.datasets import make_blobs
import numpy as np
from k_means import k_means
import matplotlib.pyplot as plt
from matplotlib import cm
import unittest


class TestKMeans(unittest.TestCase):

    def test_k_means(self):
        X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=12)
        model = k_means.KMeans(k=3)
        centroids, pts = model.fit(X)
        plt.figure()
        y_unique = np.unique(y)
        colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
        for this_y, color in zip(y_unique, colors):
            this_X = X[y == this_y]
            this_sw = X[y == this_y]
            plt.scatter(this_X[:, 0], this_X[:, 1], s=this_sw * 50, c=color,
                        alpha=0.5, edgecolor='k',
                        label="Class %s" % this_y)
        plt.legend(loc="best")
        plt.show()
        print(centroids)

if __name__ == "__main__":
    unittest.main()
