import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

df = pd.read_csv("thyroid_dataset.csv")

X = df.drop("Outlier_label",axis = 1)
y = df["Outlier_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)