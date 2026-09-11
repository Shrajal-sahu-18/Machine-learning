import seaborn as sns 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Visualize
sns.scatterplot(x = X[:,0],y = X[:,1],c = y)

#Scalling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)