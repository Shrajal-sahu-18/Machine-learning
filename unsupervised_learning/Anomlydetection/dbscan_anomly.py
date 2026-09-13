import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X,y = make_moons(
    n_samples = 500,
    noise = 0.1,
    random_state = 42
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

sns.scatterplot(x = X_scaled[:,0],y = X_scaled[:,1] )