#import module
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load data
iris = load_iris()

X = pd.DataFrame(iris.data)
X.columns = iris.feature_names
y = iris.target

#scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#import pca
from sklearn.decomposition import PCA
pca = PCA(
    n_components = 2
)
X_pca = pca.fit_transform(X_scaled)

#print variance ratio
print("explain varriance ratio :",pca.explained_variance_ratio_)
#print components
print(pca.components_)# Eigen vectors