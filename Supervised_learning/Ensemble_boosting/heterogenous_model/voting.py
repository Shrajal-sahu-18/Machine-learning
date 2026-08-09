# Voting Classifier

# Import library
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Genrate random Classification Data
X,y = make_classification(
    n_samples = 500,
    n_features = 20,
    n_informative = 5,
    n_redundant = 2
)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.3, random_state = 42
)

#Logistic regresion
lr = LogisticRegression()