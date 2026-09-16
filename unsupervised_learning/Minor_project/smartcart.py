# Import Module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.decomposition import PCA

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
df_clean = df.drop(columns = cols_to_drop)

#Outliers
cols = ["Income","Recency","Response","Age","Total_spending","Total_Children"]
# relative plots of some features - pair plots
sns.pairplot(df_clean[cols])

# Remove Outliers
print("Data size with outliers:",len(df_clean))

df_clean = df_clean[ (df_clean["Age"] < 90)]
df_clean = df_clean[(df_clean["Income"] < 600_000)]

print("Data size without outliers:",len(df_clean))

# correaltion matrix
corr = df_clean.corr(numeric_only= True)

# Correlation heatmap
plt.figure(figsize = (8,6))
sns.heatmap(
    corr,
    annot= True,
    annot_kws = {"size":6,"weight":"bold"},
    cmap = "coolwarm"
)

#Encoding
ohe =  OneHotEncoder()
cat_cols = ["Living_With","Education"]
enc_cols =  ohe.fit_transform(df_clean[cat_cols])

encoded_df = pd.DataFrame(enc_cols.toarray(),columns = ohe.get_feature_names_out(enc_cols),index = df_clean.index)

df_encoded = pd.concat([df.drop(columns = cat_cols),encoded_df],axis = 1)

X = df_encoded

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components = 3)
X_pca = pca.fit_transform(X_scaled)

#plot
sns.scatterplot(x = X_pca[:,0],y = X_pca[:,1])