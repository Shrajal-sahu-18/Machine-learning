from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression

#Genrate data
X,y= make_regression(
    n_samples = 100,
    n_features = 10,
    noise = 20,
    random_state = 42
)