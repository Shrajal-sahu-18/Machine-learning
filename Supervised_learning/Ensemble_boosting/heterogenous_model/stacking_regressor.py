from sklearn.ensemble import StackingRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

X,y = make_regression(
    n_samples = 500,
    n_features = 20,
    n_informative = 5,
    random_state = 42
)