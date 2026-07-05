import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets

#Load Dataset
df = datasets.load_iris(as_frame = True).frame

# Basic analysic
df.head()
df.isnull().sum()
df.shape

#Split Dataset
X = df.drop(["target"],axis = 1)
y = df["target"]

#Train test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y, test_size = 0.3,random_state = 42,stratify = y
) 


#Scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
svc = SVC()
svc.fit(X_train_scaled,y_train)
y_pred = svc.predict(X_test_scaled)

from sklearn.metrics import accuracy_score
print("accuracy_score:",accuracy_score(y_test,y_pred))



# Model
# svc = SVC(kernel = "linear")
# svc = SVC(kernel = "poly")
svc = SVC(kernel = "sigmoid")
svc.fit(X_train_scaled,y_train)
y_pred = svc.predict(X_test_scaled)



print("accuracy_score:",accuracy_score(y_test,y_pred))



c_vals = [0.5,1,2,3,4,5]
for cval in c_vals:
    svc = SVC(kernel = "rbf",C = cval)
    svc.fit(X_train_scaled,y_train)
    y_pred = svc.predict(X_test_scaled)
    print(cval,"accuracy_score:",accuracy_score(y_test,y_pred))