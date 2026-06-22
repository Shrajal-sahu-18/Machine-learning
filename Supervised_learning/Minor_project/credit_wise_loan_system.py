import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv("loan_approval_data.csv")
# print(df.head())

df.head()
df.info()
df.isnull().sum()
df.describe()

# Handle missing value
categorical_cols = df.select_dtypes(include = "object").columns
numerical_cols = df.select_dtypes(include = "float64").columns 

categorical_cols.size + numerical_cols.size 

from sklearn.impute import SimpleImputer