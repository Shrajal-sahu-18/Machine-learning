import pandas as pd
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#Load data
heart_df = pd.read_csv("heart.csv")
heart_df.head()

#split dataset into parts
X = heart_df.drop(["target"],axis = 1)
y = heart_df["target"]

#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y ,test_size = 0.2,random_state = 42
)

#Scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled =  scaler.transform(X_test)

#train the model#k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_scaled,y_train)

y_pred = knn_classifier.predict(X_test_scaled)

#Evaluation
print("precision_score:",precision_score(y_test,y_pred))
print("accuracy_score:",accuracy_score(y_test,y_pred))
print("recall_score:",recall_score(y_test,y_pred))


#k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled,y_train)

y_pred = knn_classifier.predict(X_test_scaled)

#Evaluation
print("precision_score:",precision_score(y_test,y_pred))
print("accuracy_score:",accuracy_score(y_test,y_pred))
print("recall_score:",recall_score(y_test,y_pred))



#k = 7
knn_classifier = KNeighborsClassifier(n_neighbors = 7)
knn_classifier.fit(X_train_scaled,y_train)

y_pred = knn_classifier.predict(X_test_scaled)

#Evaluation
print("precision_score:",precision_score(y_test,y_pred))
print("accuracy_score:",accuracy_score(y_test,y_pred))
print("recall_score:",recall_score(y_test,y_pred))

#Cross validation for hyperparameter tuning using GridsearchCV
from sklearn.model_selection import GridSearchCV
classifier =  KNeighborsClassifier()
param_grid = {"n_neighbors":[3,5,7,9]}

classifierCV = GridSearchCV(
    classifier,
    param_grid,
    cv = 5,
    scoring = "recall"
    
)
classifierCV.fit(X_train_scaled,y_train)
y_pred = classifierCV.predict(X_test_scaled)

#Evaluation
print("precision_score:",precision_score(y_test,y_pred))
print("accuracy_score:",accuracy_score(y_test,y_pred))
print("recall_score:",recall_score(y_test,y_pred))

result = pd.DataFrame(classifierCV.cv_results_)

print(result[["param_n_neighbors","mean_test_score"]])
print(classifierCV.best_params_)

#Pipeline
from sklearn.pipeline import Pipeline
X_train,X_test,y_train,y_test = train_test_split(
    X,y ,test_size = 0.2,random_state = 42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()), ('knn',  KNeighborsClassifier())
])

param_grid = {"knn__n_neighbors":[3,5,7,9]}
classifierCV = GridSearchCV(
    pipeline,
    param_grid,
    cv = 5,
    scoring = "recall"
    
)
classifierCV.fit(X_train,y_train)
y_pred = classifierCV.predict(X_test)