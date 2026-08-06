from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report

#genrate Data
X,y = make_classification(
    n_samples = 500,
    n_features = 20,
    n_informative = 10,
    n_redundant = 2,
    random_state = 42
)