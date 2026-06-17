import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

#Load Dataset
df = pd.read_csv("employee_turnover.csv")

#Split Dataset into two parts
X = df.drop(["Employee_Turnover"],axis = 1)
y = df["Employee_Turnover"]