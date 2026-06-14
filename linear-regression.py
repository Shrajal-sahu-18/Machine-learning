import pandas  as pd 
import seaborn as sns
import matplotlib.pyplot as plt

insurance_data = pd.read_csv("insurance.csv")
# print(insurance_data)

sns.scatterplot(x = insurance_data["bmi"],y = insurance_data["charges"],hue = insurance_data["smoker"])
# plt.show()

#split dataset into X and y

X = insurance_data.drop(columns = ["charges","region"])
y = insurance_data["charges"]

X["sex"] = X["sex"].map({"female":1,"male":0})
X["smoker"] = X["smoker"].map({"yes":1,"no":0})

#Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
# print(y_pred)
# print(y_test)



#Evaluate
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("r2_score:",r2)
n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - ((1 -r2) * (n - 1)) / (n - p - 1)
print("adjusted_r2:",adjusted_r2)


#remove charges 
X = insurance_data.drop(columns = ["charges"])
y = insurance_data["charges"]

X = pd.get_dummies(X,columns = ["region"],drop_first = True,dtype = int)
X["sex"] = X["sex"].map({"female":1,"male":0})  # convert female,male,1,0
X["smoker"] = X["smoker"].map({"yes":1,"no":0})