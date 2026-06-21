import numpy as np
class LinearRegression:
    def __init__(self,learning_rate = 0.01,n_iter = 1000):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter
    
    def fit(self,X,y):
        m,n = X.shape #(number of row number of column = sample,features)
    
        #sTEP - 1 # intilize the parameter
        self.bias = 0
        self.weights = np.zeros(n)
        #Step - 2
        for i in range(self.n_iter):
            y_pred = self.bias + np.dot(X,self.weights)


