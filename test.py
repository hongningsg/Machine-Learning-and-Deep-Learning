import numpy as np
import Logistic_Model.LogisticRegression as L
intercept = np.ones((5, 1))

X = np.random.rand(10, 5)
Y = np.array([1,0,1,0,1,0,1,0,1,1]).T

mymodel = L()
W = mymodel.fit(X, Y)
print(W.shape)

print("########################################")
print(np.dot(X,W))
