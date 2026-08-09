# Voting Regressor
from sklearn.datasets import make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR



X,y = make_regression(
    n_samples = 500,
    n_features  = 20,
    n_informative = 5,
    random_state = 42
)