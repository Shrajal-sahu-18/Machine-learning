import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

#Load Dataset
df = pd.read_csv("employee_turnover.csv")

#Split Dataset into two parts
X = df.drop(["Employee_Turnover"],axis = 1)
y = df["Employee_Turnover"]

#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X , y, test_size = 0.2 , random_state = 42
)

# Baseline Logistic Regression
lr =LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

#l1 Regularization Lasso
l1 = LogisticRegression(penalty = "l1",solver = "liblinear", C = 0.5)
l1.fit(X_train,y_train)