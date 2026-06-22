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

num_imp = SimpleImputer(strategy = "mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

categorical_imp = SimpleImputer(strategy = "most_frequent")
df[categorical_cols] = categorical_imp.fit_transform(df[categorical_cols])

#EDA(Exploratory Data Analysis)