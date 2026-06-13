# import library
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from sklearn.preprocessing import StandardScaler

#Load data

df = pd.read_csv("HousePricePrediction.csv")

#Basic anaylsis
df.shape
df.head()
df.info()
df.describe()



# Data Preprocessing - Data cleaning

# Drop Id as it doesn't contribute in price
df.drop(['Id'], axis=1, inplace=True)

# Replacing SalePrice empty values with their mean values
df['SalePrice'] = df['SalePrice'].fillna(df['SalePrice'].mean()) 