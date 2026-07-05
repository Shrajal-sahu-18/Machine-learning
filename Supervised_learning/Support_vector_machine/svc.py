import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets

#Load Dataset
df = datasets.load_iris(as_frame = True).frame

# Basic analysic
df.head()
df.isnull().sum()
df.shape

#Split Dataset
X = df.drop(["target"],axis = 1)
y = df["target"]

#Train test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y, test_size = 0.3,random_state = 42,stratify = y
) 