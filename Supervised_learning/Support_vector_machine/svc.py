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
