import pandas as pd
df = pd.read_csv("iris.csv")
df.head()

# convert 3 species into 0, 1, 2
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])
print(df)

#Train test split
from sklearn.model_selection import train_test_split

X = df.drop(["Species"],axis = 1)
y = df["Species"]