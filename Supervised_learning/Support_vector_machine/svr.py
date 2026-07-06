from sklearn.svm import SVR
import pandas as pd
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

#load datasets
df = datasets.load_diabetes(as_frame = True).frame
df.head()

#Split dataset
X = df.drop(["target"],axis = 1)
y = df["target"]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# scalling 
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()

#Fit Model
svr = SVR()
svr.fit(X_train,y_train_scaled)

# Test Model
y_pred_scaled = svr.predict(X_test)

#R2 Score
print(r2_score(y_test_scaled,y_pred_scaled))

# Kernel change 
svr = SVR(kernel = "linear")
svr.fit(X_train,y_train_scaled)
y_pred_scaled = svr.predict(X_test)
print(r2_score(y_test_scaled,y_pred_scaled))

# Hyperparameter tunning using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {"C":[1,2,5,10,50,100],
              "kernel":["rbf","linear"],
              "epsilon":[0.01,0.1,0.2,0.3,0.5]
}

#Fit the model
svr = SVR()
grid_search = GridSearchCV(svr,param_grid,scoring = "r2",cv = 5)
grid_search.fit(X_train,y_train_scaled)