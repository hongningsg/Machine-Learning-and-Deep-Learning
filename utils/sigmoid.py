from numpy import exp


def sigmoid(z):
    return 1/(1+exp(-z))


def step_function(threshold, h):
    if h > threshold:
        return 1
    return 0
