import pandas as pd
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
heart_df = pd.read_csv("heart.csv")


heart_df.head()
heart_df.columns
heart_df.info()
heart_df["target"].nunique()

#Split dataset into two parts
X = heart_df.drop("target",axis = 1)
y = heart_df["target"]
X.head()
y.head()

#tain test split
X_train, X_test,y_train,y_test = train_test_split(
    X,y ,test_size = 0.2 , random_state = 42
)
y_train[y_train == 1] #133
y_train[y_train == 0] #109
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test,y_pred))
print("precision:", precision_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))