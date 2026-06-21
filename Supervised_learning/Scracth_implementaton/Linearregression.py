import numpy as np
class LinearRegression:
    def __init__(self,learning_rate = 0.01,n_iter = 1000):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter
