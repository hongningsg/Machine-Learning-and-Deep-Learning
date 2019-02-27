from utils import normalize
import unittest
import numpy as np

class TestUtils(unittest.TestCase):

    def test_normalize(self):
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        normalized_X = normalize(X)
        np.testing.assert_array_almost_equal(normalized_X, np.array([[-0.63299316, -0.04124145,  0.55051026],
                                                                    [ 2.36700684,  2.95875855,  3.55051026],
                                                                    [ 5.36700684,  5.95875855,  6.55051026]]))


if __name__ == '__main__':
    unittest.main()
