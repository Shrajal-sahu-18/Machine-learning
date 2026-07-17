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
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2,random_state = 42
)
#import gradientboost regresor
from sklearn.ensemble import GradientBoostingRegressor

ghr = GradientBoostingRegressor(
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 3,
    subsample = 0.8,
    random_state = 42
)

ghr.fit(X_train,y_train)
y_pred = ghr.predict(X_test)
print("r2:",r2_score(y_test,y_pred))


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score