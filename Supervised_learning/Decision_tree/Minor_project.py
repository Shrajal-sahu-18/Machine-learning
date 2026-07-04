import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix

#Load Dataset
df = pd.read_csv("shop_smart_ecommerce.csv")

#Split Dataset
X = df.drop(columns = ["Revenue"])
y = df["Revenue"]

#Divide category
num_cols = X.select_dtypes(include =["int64","float64"]).columns
cate_cols = X.select_dtypes(include =["category","object","bool"]).columns

X_train,X_test,y_train,y_test = train_test_split(
    X,y , test_size = 0.3 , random_state = 42
)