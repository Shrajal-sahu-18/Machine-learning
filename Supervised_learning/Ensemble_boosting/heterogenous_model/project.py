# Import module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, classification_report,accuracy_score 

#Load Dataset
df = pd.read_csv("novagen_dataset.csv")

#Split dataset
X = df.drop(columns = ["Target"])
y = df["Target"]

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X , y ,test_size = 0.3 , random_state = 42
)

#Standardscaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with regularization
log_reg = LogisticRegression(
    penalty = "l2",
    solver = "liblinear",
    max_iter = 1000
) 