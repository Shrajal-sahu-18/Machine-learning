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

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score
print("accuracy_score:",accuracy_score(y_test,y_pred))

from sklearn.tree import plot_tree
plt.figure(figsize = (18,10))

plot_tree(
    model,
    feature_names = X.columns,
    class_names = ["Died","Survived"],
    filled = True,
    max_depth = 2
)
plt.tight_layout(),
plt.show()


#preprunning
max_depths = [2,3,4,5,6,7,8,9,10]
for depth in max_depths:
    model = DecisionTreeClassifier(max_depth = depth)
    model.fit(X_train,y_train)
    acc = model.score(X_test,y_test) # Accuracy score
    print(f"for depth{depth}, accuracy{acc}")
    if depth == 4:
        plt.figure(figsize = (18,10))

        plot_tree(
            model,
            feature_names = X.columns,
            class_names = ["Died","Survived"],
            filled = True
            # max_depth = 4
        )
        plt.tight_layout(),
        plt.show()
min_samples_splits = [10,15,20,25,30]
for split in min_samples_splits:
    model = DecisionTreeClassifier(max_depth = 6,min_samples_split = split) # Split tabhi hoga jab algorithm ko koi usefull split mile ahar information gain or gini impurity improve nhi ho rahi hai to split ruk jayega
    model.fit(X_train,y_train)
    acc = model.score(X_test,y_test) # Accuracy score
    print(f"for split {split}, accuracy{acc}")

#Decision Tree With post-prunning