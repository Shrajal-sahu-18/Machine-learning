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

               #Step 3  gradients
            db = (1/m) * np.sum(y_pred - y)
            dw = (1/m) * np.dot(X.T , (y_pred - y))

             #Step 4 convergence theorem - param update
            self.bias = self.bias - self.lr * db # -= self.lr + db
            self.weights = self.weights - self.lr * dw # -=self.lr + dw
    
        
    def predict(self,X):
        y_pred = self.bias + np.dot(X,self.weights)
        return y_pred

        
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

