import seaborn as sns 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

#Load dataset
iris = load_iris()

X = iris.data
y = iris.target