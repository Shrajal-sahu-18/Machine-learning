import pandas as pd
df = pd.read_csv("iris.csv")
df.head()

# convert 3 species into 0, 1, 2
from sklearn.preprocessing import LabelEncoder