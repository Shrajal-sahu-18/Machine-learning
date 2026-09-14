import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("thyroid_dataset.csv")

X = df.drop("Outlier_label",axis = 1)
y = df["Outlier_label"]