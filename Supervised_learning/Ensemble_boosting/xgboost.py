#install xgboost
# xgboostclassifier
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

#Genrate data for classification model
X,y = make_classification(
    n_samples = 500,
    n_features = 20,
    n_informative = 10,
    n_redundant = 2,
    random_state = 42
    
)

# train test split 
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.3,random_state = 42
)

#xgbboost classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators = 100,
    max_depth = 3,
    learning_rate = 0.3,
    eval_metric = "logloss",
    random_state = 42
)

#fit the model
xgb_clf.fit(X_train,y_train)

#Test the model
y_pred = xgb_clf.predict(X_test)

#test the accuracy 
print("Accuracy_Score",accuracy_score(y_test,y_pred))

#regressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

X,y = make_regression(
    n_samples = 1000,
    n_features = 20,
    n_informative = 10,
   
)
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.3,random_state = 42
)