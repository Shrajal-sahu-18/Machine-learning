# Import module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, classification_report,accuracy_score 

#Load Dataset
df = pd.read_csv("novagen_dataset.csv")

#Split dataset
X = df.drop(columns = ["Target"])
y = df["Target"]