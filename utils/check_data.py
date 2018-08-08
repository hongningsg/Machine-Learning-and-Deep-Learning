import numpy as np


def check_data(X, y):
    X = check_X(X)
    y, c = check_y(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError('dim 0 of X and y are not aligned')
    print("X dim:", X.shape)
    print("y dim:", y.shape)
    print("y classes:", c)
    return X, y, c

def check_X(X):
    if not isinstance(X, np.ndarray):
        raise TypeError('X must be numpy.ndarray')
    # convert (150, ) to (1, 150)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    return X

def check_y(y):
    if not isinstance(y, np.ndarray):
        raise TypeError('y must be numpy.ndarray')
    # force y clsses to be 0, 1, 2, .. etc
    unique, y = np.unique(y, return_inverse=True)
    c = unique.shape[0] # nb of classes
    if c < 2:
        raise ValueError("number of classes must be >= 2")
    # convert (150, ) to (150, 1)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    return y, c

# X = np.array([[0.7, 0.1, 0.9],
#               [0.1, 0.8, 0.1],
#               [0.8, 0.2, 0.8]])
# y = np.array([1, 0, 1]).reshape(3, 1)
#
# print(np.unique(y, return_inverse=True))
# check_data(X, y)
