#import module
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#Load data
iris = load_iris()
X = iris.data
y = iris.target

#Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)