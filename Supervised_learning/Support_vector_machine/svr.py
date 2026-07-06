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