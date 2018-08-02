import numpy as np

"""
X: (m, n)
w: (1, n)
b: (1, 1)
y: (1, m)
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# w shape: (1,
def linear(X, w, b):
    return np.dot(w, X) + b

X = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1, 2], np.newaxis])

print(X.shape)
print(w.shape)
# z = linear(X, w, b)

