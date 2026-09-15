# Import Module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv("smartcart_customers.csv")

#Check null value
df.isnull().sum()

#Handle missing value
df["Income"] = df["Income"].fillna(df["Income"].median())

df.isnull().sum()

# Feature Engineering
df.head()
df.columns

# Create New Column Age
df["Age"] = 2026 - df["Year_Birth"]

# Customer joining days
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"],dayfirst = True)
refrence_date = df["Dt_Customer"].max()

# Create new columns customers tenure days
df["Customer_tenure_days"] = (refrence_date - df["Dt_Customer"]).dt.days