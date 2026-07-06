from sklearn.svm import SVR
import pandas as pd
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

#load datasets
df = datasets.load_diabetes(as_frame = True).frame
df.head()