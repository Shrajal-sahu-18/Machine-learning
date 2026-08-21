from  sklearn.ensemble import StackingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X,y = make_classification(
    n_samples = 500,
    n_features = 20,
    n_informative = 5,
    n_redundant = 3
)


X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.3,random_state = 42
)

lr = LogisticRegression()
svc = SVC()
dtc = DecisionTreeClassifier(max_depth = 4)