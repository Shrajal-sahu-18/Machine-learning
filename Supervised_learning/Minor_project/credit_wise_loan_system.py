import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


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