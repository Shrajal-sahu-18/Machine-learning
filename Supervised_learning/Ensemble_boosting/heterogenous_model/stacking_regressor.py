from sklearn.ensemble import StackingRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X,y = make_regression(
    n_samples = 500,
    n_features = 20,
    n_informative = 5,
    random_state = 42
)

X_train,X_test,y_train,y_test = train_test_split(
    X,y, test_size = 0.3, random_state = 42
)

lr = LinearRegression()
svr = SVR()
dtr = DecisionTreeRegressor()



sr = StackingRegressor(
    estimators = [
        ("lr",lr),
        ("svr",svr),
        ("dtr",dtr)
    ],
    cv = 5
    
)


sr.fit(X_train,y_train)


y_pred = sr.predict(X_test)