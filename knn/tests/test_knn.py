import unittest
import numpy as np
from knn import knn
from sklearn.datasets import load_iris


class TestKNearestNeighbor(unittest.TestCase):

    def test_distance(self):
        x1 = np.array([[1, 3, 5]])
        x2 = np.array([[2, 3, 1]])
        model1 = knn.KNNClassifier(k=1)
        self.assertEqual(model1._distance(x1, x2), np.power(17, 0.5))
        model2 = knn.KNNClassifier(k=2, norm=1)
        self.assertEqual(model2._distance(x1, x2), 5)

    def test_kNNClassifier(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        model = knn.KNNClassifier(k=3)
        model.fit(X[:100], y[:100])
        preds = model.predict(X[100:])
        print(preds)
        print("True y:", y[100:])


if __name__ == '__main__':
    unittest.main()
