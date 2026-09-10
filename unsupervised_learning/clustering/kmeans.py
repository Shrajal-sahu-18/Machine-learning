import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X,y = make_blobs(
    n_samples = 1000,
    n_features = 2,
    centers = 4,
    random_state = 42
)

#scatter plot
sns.scatterplot(x = X[:,0],y = X[:,1])

# kmeans model
k = 4
kmeans = KMeans(
    n_clusters = k,
    random_state = 42
)