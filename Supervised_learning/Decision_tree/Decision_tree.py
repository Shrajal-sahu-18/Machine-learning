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