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

# Visulize the data
sns.scatterplot(x = X_scaled[:,0],y = X_scaled[:,2],c = y)

#Dendrogram

from scipy.cluster.hierarchy import linkage,dendrogram