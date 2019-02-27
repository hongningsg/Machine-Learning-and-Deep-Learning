from sklearn.datasets.samples_generator import make_blobs
from svm import svm
import numpy as np
import unittest
import matplotlib.pyplot as plt

class Testsvm(unittest.TestCase):
    # 100 pos + 100 neg
    X, y = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=0.5, random_state=12)
    y_unique, y_counts = np.unique(y, return_counts=True)
    # plt.scatter(X[:, 0], X[:, 1], marker='o',c=y) # c means colors based on y
    # plt.show()

    def test_split_data(self):
        model = svm.svm()
        data_dict = model.split_data(self.X, self.y)
        for v in data_dict.values():
            self.assertEqual(len(v), 100)

    def test_gram_matrix(self):
        model = svm.svm()
        X = np.array([[1, 3],
                      [2, 1],
                      [0, 1]])
        y = np.array([1, 1, -1]).reshape(3, 1)
        G = np.array([[10, 5, -3],
                      [5, 5, -1],
                      [-3, -1, 1]])

        Gram_matrix = model._gram_matrix(X, y)
        self.assertTrue(np.array_equal(G, Gram_matrix))

if __name__ == "__main__":
    unittest.main()
