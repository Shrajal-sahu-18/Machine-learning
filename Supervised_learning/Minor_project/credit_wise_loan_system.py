import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,confusion_matrix,recall_score,f1_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("loan_approval_data.csv")
# print(df.head())

df.head()
df.info()
df.isnull().sum()
df.describe()

# Handle missing value
categorical_cols = df.select_dtypes(include = "object").columns
numerical_cols = df.select_dtypes(include = "float64").columns 

categorical_cols.size + numerical_cols.size 

from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy = "mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

categorical_imp = SimpleImputer(strategy = "most_frequent")
df[categorical_cols] = categorical_imp.fit_transform(df[categorical_cols])

#EDA(Exploratory Data Analysis)

# how balanced our classes are
classes_count = df["Loan_Approved"].value_counts()
plt.pie(classes_count , labels = ["No","Yes"], autopct = "%1.1f%%")
plt.title("Is loan approved or not")

gender_cnt = df["Gender"].value_counts()
ax =sns.barplot(gender_cnt)
ax.bar_label(ax.containers[0])


edu_cnt = df["Education_Level"].value_counts()
ax = sns.barplot(edu_cnt)
ax.bar_label(ax.containers[0])

emp_cunt = df["Employment_Status"].value_counts()
ax = sns.barplot(emp_cunt)
ax.bar_label(ax.containers[0])



martial_cnt = df["Marital_Status"].value_counts()
ax = sns.barplot(martial_cnt)
ax.bar_label(ax.containers[0])

purpose_cnt = df["Loan_Purpose"].value_counts()
ax = sns.barplot(purpose_cnt)
ax.bar_label(ax.containers[0])


property_area_cnt =df["Property_Area"].value_counts()
ax = sns.barplot(property_area_cnt)
ax.bar_label(ax.containers[0])


empl_cate_cnt = df["Employer_Category"].value_counts()
ax = sns.barplot(empl_cate_cnt)
ax.bar_label(ax.containers[0])



#Analyze Income
sns.histplot(
    data = df,
    x = "Applicant_Income",
    bins = 20
    
)


sns.histplot(
    data = df,
    x = "Coapplicant_Income",
    bins = 20
)


#Outliers - box_plot
sns.boxplot(
    data = df,
    x = "Loan_Approved",
    y = "Applicant_Income"
)

fig,axes = plt.subplots(2,3)

sns.boxplot(ax = axes[0,0], data = df,x = "Loan_Approved",y = "Applicant_Income")
sns.boxplot(ax = axes[0,1], data = df,x = "Loan_Approved", y = "Credit_Score")
sns.boxplot(ax = axes[0,2],data = df,x = "Loan_Approved",y = "Age")
sns.boxplot(ax = axes[1,0], data = df, x = "Loan_Approved", y = "DTI_Ratio")
sns.boxplot(ax = axes[1,1], data = df , x = "Loan_Approved", y = "Savings")
sns.boxplot(ax = axes[1,2],data = df,x = "Loan_Approved",y = "Loan_Amount")

plt.tight_layout()

#credit score with loan approved
sns.histplot(
    data = df,
    x = "Credit_Score",
    hue = "Loan_Approved",
    bins = 20,
    multiple = "dodge" 
    # side by side dodge ka mtlb alag alag category ke bars ko alag alag dikhana
)


sns.histplot(
    data = df,
    x = "Applicant_Income",
    hue = "Loan_Approved",
    multiple = "dodge",
    bins = 20
)

#Remove_applicant_id
df = df.drop("Applicant_ID",axis = 1)

#Encoding 
df.head()
df.columns
df.info()


#labelEncoder
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

#oNEhOTeNCODING
cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
ohe = OneHotEncoder(drop = "first",sparse_output = False,handle_unknown = "ignore")
encoded = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded,columns = ohe.get_feature_names_out(cols),index = df.index )
df = pd.concat([df.drop(columns = cols),encoded_df],axis = 1)


#correlation heat map
num_cols = df.select_dtypes(include = "number")
corr_matrix = num_cols.corr()
plt.figure(figsize = (15,8))
sns.heatmap(
    corr_matrix,
    annot = True,
    fmt = ".2f",
    cmap = "coolwarm"
)
num_cols.corr()["Loan_Approved"].sort_values(ascending = False)
# corr_matrix

#split dataset 
X = df.drop(["Loan_Approved"],axis = 1)
y = df["Loan_Approved"]

#train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y, test_size = 0.2 , random_state  =42
)

#Standard Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train linearregression model
logs_model = LogisticRegression()
logs_model.fit(X_train_scaled,y_train)
y_pred = logs_model.predict(X_test_scaled)

# Evaluation
print("precision score:",precision_score(y_test,y_pred))
print("recall score:",recall_score(y_test,y_pred))
print("f1 score:",f1_score(y_test,y_pred))
print("confusion matrix:",confusion_matrix(y_test,y_pred))
print("Accuracy score:",accuracy_score(y_test,y_pred))


#knn model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)