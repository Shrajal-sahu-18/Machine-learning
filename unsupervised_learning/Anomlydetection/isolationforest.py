import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

df = pd.read_csv("thyroid_dataset.csv")

X = df.drop("Outlier_label",axis = 1)
y = df["Outlier_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#clf = classifier
clf = IsolationForest(
    n_estimators = 200,
    contamination= 0.036,
    # contamination='auto',
      # auto ka mtlb reseach paper ke under jitne value set set hai 
    random_state = 42
)

labels = clf.fit_predict(X_scaled)

# visulize
from sklearn.decomposition import PCA
pca = PCA(
    n_components = 2
)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize = (8,6))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c = labels
)
plt.xlabel("PCA1")
plt.ylabel("PCA2")