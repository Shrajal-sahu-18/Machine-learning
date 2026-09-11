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

#Optional - dimensionality reduction using PCA
pca = PCA(n_components = 2)
pca_data = pca.fit_transform(X_scaled)

#Elbow method
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit_predict(pca_data)
    wcss.append(kmeans.inertia_)

sns.lineplot(x = range(1,11),y = wcss,marker = 'o')

#kmeans
kmeans = KMeans(n_clusters = 3,random_state = 42)
labels = kmeans.fit_predict(pca_data)
sns.scatterplot(x = pca_data[:,0],y = pca_data[:,1],c= labels)