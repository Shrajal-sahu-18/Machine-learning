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

# Total Spending new column
df["Total_spending"] =  df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

#Total Children New column
# Children
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

#Education
df["Education"].value_counts()

# undergraduate,postgraduate,graduate
df["Education"] = df["Education"].replace({
    "Basic":"Undergraduate","2n Cycle":"Undergraduate",
    "Graduation":"Graduate",
    "PhD":"Postgraduate","Master":"Postgraduate"
})


df["Education"].value_counts()

# Martial Status
df["Marital_Status"].value_counts()

#Martial Status
df["Living_With"]  = df["Marital_Status"].replace({
    "Married":"Partner","Together":"Partner",
    "Single":"Alone",""
    "Single":"Alone","Divorced":"Alone",
    "Widow":"Alone","Absurd":"Alone","YOLO":"Alone"
})

df["Living_With"].value_counts()

# Drop columns
cols = ["ID","Year_Birth","Marital_Status","Kidhome","Teenhome","Dt_Customer"]
spending_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]

cols_to_drop = cols + spending_cols