import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

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

labels = kmeans.fit_predict(X)

(sns.scatterplot(x = X[:,0],y = X[:,1],c = labels))

#chossing k value with Elbow method
wcss = []
for k in range(1,21):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
print(wcss)

sns.lineplot(x = range(1,21),y = wcss,marker = 'o')

#kneelocator
knee = KneeLocator(range(1,21),wcss,curve = "convex",direction = "decreasing")
print(knee.knee)
print(knee.elbow)

#silhouette score
ss = []
for k in range(2,21):
    kmeans = KMeans(n_clusters = k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X,labels)
    ss.append(score)