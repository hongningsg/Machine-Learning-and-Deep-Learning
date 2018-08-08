import numpy as np
from utils import check_data

class BernoulliNB():
    def __init__(self, smooth=True):
        self.smooth = smooth
        self.model = {}
        self.prob = []

    def _data_processing(self, X, Y):
        X = np.logical_and(check_data.check_X(X), 1).astype(int)
        Y, c = check_data.check_y(Y)
        merge = np.concatenate((X, Y), axis=1)
        y_unique, y_counts = np.unique(Y, return_counts=True)
        classes = dict()
        for i in range(len(y_unique)):
            classes[y_unique[i]] = np.sum(merge[np.where(merge[:,-1] == y_unique[i])], axis = 0)
            classes[y_unique[i]][-1] = y_counts[i]
        if self.smooth:
            classes = self._smooting(classes, y_unique)
        return classes

    def _smooting(self, classes, y_u):
        for i in range(len(y_u)):
            classes[y_u[i]] += 1
            classes[y_u[i]][-1] += 1
        return classes

    def fit(self, X, Y):
        classes = self._data_processing(X, Y)
        for key in classes:
            total = classes[key][-1]
            classes[key] = classes[key]/total
            classes[key][-1] = total - 2
        self.model = classes

    def predict(self, X):
        X = np.logical_and(check_data.check_X(X), 1).astype(int)
        assert len(self.model) > 0, "Model has not fitted before predict!"
        y = np.array([0]*X.shape[0]).T
        iteration = 0
        pred_data = dict()
        for key in self.model:
            pred_model = self.model[key][:-1].T
            pred_result = np.prod(pred_model * X + (1 - pred_model) * (1 - X), axis = 1) * self.model[key][-1]
            pred_data[key] = np.column_stack((pred_result, np.array([key]*X.shape[0]).T))
            if iteration > 0:
                y = np.where(pred_data[key][:,:-1] > y[:,:-1], pred_data[key], y)
            else:
                y = pred_data[key]
            iteration += 1
        self.prob = y[:,:-1]/X.shape[0]
        return y[:,-1:].astype(int)




if __name__ == "__main__":
    clf = BernoulliNB()
    X = np.array([[1,0,1],[1,1,1],[0,0,1]])
    Y = np.array([1, 2, 0])
    x = np.concatenate((np.array([[2],[3]]),np.ones((2,1),dtype=int)), axis = 1)
    y = np.array([[1,0],[4,0]])
    clf.fit(X, Y)
    print(clf.predict(X))
    print(clf.prob)
