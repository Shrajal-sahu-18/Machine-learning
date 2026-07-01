import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

titanic = sns.load_dataset("titanic")
titanic.head(5)
titanic.info()
titanic.isnull().sum()

feature = ["pclass","sex","fare","embarked","age"]
target = ["survived"]

#Missing value
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(strategy = "median")
titanic[["age"]] = imp_median.fit_transform(titanic[["age"]]) 
most_freq = SimpleImputer(strategy = "most_frequent")
titanic[["embarked"]] = most_freq.fit_transform(titanic[["embarked"]]) 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
titanic["sex"] = le.fit_transform(titanic["sex"])
titanic["embarked"] = le.fit_transform(titanic["embarked"])


X = titanic[feature]
y = titanic[target]

X_train,X_test,y_train,y_test = train_test_split(
    X,y , test_size = 0.2,random_state = 42
)

# Decision tree classifier - no prunning
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,y_train)