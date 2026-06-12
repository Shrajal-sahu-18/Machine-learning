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