import pandas as pd
from  sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#Load datset
heart_df = pd.read_csv("heart.csv")

heart_df.head()

#Split dataset into two parts
X = heart_df.drop(["target"] , axis = 1)
y = heart_df["target"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y , test_size = 0.2,random_state = 42
)

#naive bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train,y_train)

#test model
y_pred = gnb_model.predict(X_test)
print(y_pred)
print(y_test)