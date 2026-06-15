from sklearn.linear_model import Lasso #$linear_model
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
from sklearn.model_selection import train_test_split

#load data
insurance_data = pd.read_csv("insurance.csv")

#interaction feautre
X = insurance_data.drop(columns = ["charges"])
y = insurance_data["charges"]

#One hot encoding
X = pd.get_dummies(X,columns = ["region"],drop_first = True,dtype = int)
# add new feautre
X["sex"] = X["sex"].map({"female":1,"male":0})  # convert female,male,1,0
X["smoker"] = X["smoker"].map({"yes":1,"no":0})

X_train,X_test,y_train,y_test = train_test_split(
    X,y, test_size = 0.2, random_state = 42
)

lasso_model = Lasso(alpha = 0.5)
lasso_model.fit(X_train,y_train)

y_pred = lasso_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean sqared error:",mse)