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

# Drop records with null values
df = df.dropna()

df.isnull().sum()
df.head()

# Data Preprocessing - Encoding 
cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
df = pd.get_dummies(df, columns=cols, drop_first=True)


# Train-Test Split
X = df.drop(['SalePrice'], axis=1)
Y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)



# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
print("r2_score:",r2_score(y_test,y_pred))
print("mae:",mean_absolute_error(y_test,y_pred))
print(" root mean sqared error:",np.sqrt(mean_squared_error(y_test,y_pred)))
print(mean_absolute_percentage_error(y_test, y_pred))

# Feature Scaling - to try & improve baseline performance
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)