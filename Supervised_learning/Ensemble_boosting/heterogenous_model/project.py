# Import module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, classification_report,accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


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

# Fit the data
log_reg.fit(X_train_scaled,y_train)

# Test data 
y_pred_lr = log_reg.predict(X_test_scaled)

# Evaluation metrics
print("recall_score",recall_score(y_test,y_pred_lr))
print("Classification_report",classification_report(y_test,y_pred_lr))
print("Accuracy_Score",accuracy_score(y_test,y_pred_lr))


# Model 2 -Knn
knn_model = KNeighborsClassifier(
    n_neighbors = 3,
    metric = "euclidean"
)
knn_model.fit(X_train_scaled,y_train)
y_pred = knn_model.predict(X_test_scaled)

print("Accuracy_score",accuracy_score(y_test,y_pred))
print("Classification_report",classification_report(y_test,y_pred))
print("recall_score",recall_score(y_test,y_pred))

# Model 3 - RandomForestclassifier
rfc = RandomForestClassifier(
    n_estimators = 200,
    max_depth = None,
    random_state = 42
)
rfc.fit(X_train , y_train)
y_pred = rfc.predict(X_test)

print("RandomForestClassifier Accuracy",accuracy_score(y_test,y_pred))
print("RandomForestClassifier Recall",recall_score(y_test,y_pred))
print("RandomForestClassifier Classification Report",classification_report(y_test,y_pred))

# Model 4 - Gradient Boosting Classifier
gb = GradientBoostingClassifier(
    n_estimators = 150,
    learning_rate = 0.1,
    max_depth = 3,
    random_state = 42
    
)
gb.fit(X_train,y_train)