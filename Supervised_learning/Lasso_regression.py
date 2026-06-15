from sklearn.linear_model import Lasso #$linear_model
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
from sklearn.model_selection import train_test_split

#load data
insurance_data = pd.read_csv("insurance.csv")

#interaction feautre
X = insurance_data.drop(columns = ["charges"])
y = insurance_data["charges"]