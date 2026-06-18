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