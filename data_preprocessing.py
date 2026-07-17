# data_preprocessing.py
# Split verbatim from credit_approver.ipynb (cells 0-31: data loading, cleaning, EDA, encoding)
# No code was changed from the original notebook cells.

import pandas as pd
import numpy as mp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("loan_approval_data.csv")
df.head()

df.info()
df.isnull().sum()
df.describe()

#handle missing values
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["float64"]).columns # or use number instead of float64

categorical_cols

numerical_cols

categorical_cols.size + numerical_cols.size

# use sklearn simpleimputer

from sklearn.impute import SimpleImputer
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

df.head()

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

df.head()

df.isnull().sum()

#yazoo! we've removed all null values in the dataset
# exploratory data analysis (EDA) - we want to look at how balanced our classes are

classes_count = df["Loan_Approved"].value_counts()
plt.pie(classes_count, labels=["No","Yes"], autopct="%1.1f%%")
plt.title("Is loan approved?")
plt.show()

# analyse categories

gender_cnt = df["Gender"].value_counts()
ax = sns.barplot(gender_cnt)
ax.bar_label(ax.containers[0])

edu_cnt = df["Education_Level"].value_counts()
lx = sns.barplot(edu_cnt)
lx.bar_label(lx.containers[0])

# analyse income

sns.histplot(
    data=df,
    x= "Applicant_Income",
    bins=20
)

sns.histplot(
    data=df,
    x= "Coapplicant_Income",
    bins=20
)

# outliers //box plots
fig, axes = plt.subplots(2,2)
sns.boxplot(ax=axes[0,0], data=df, x="Loan_Approved", y="Applicant_Income")
sns.boxplot(ax=axes[0,1], data=df, x="Loan_Approved", y="Credit_Score")
sns.boxplot(ax=axes[1,0], data=df, x="Loan_Approved", y="DTI_Ratio")
sns.boxplot(ax=axes[1,1], data=df, x="Loan_Approved", y="Savings")
plt.tight_layout()

sns.boxplot(data=df, x="Loan_Approved", y="Age")

sns.boxplot(data=df, x="Loan_Approved", y="Loan_Amount")

# 2 types of outliers
# 1) impossible/wrong, dependants -> -1 or age -> 10, 150 etc
# 2) meaningless 

# credit score with loan approval
sns.histplot(data=df, x="Credit_Score", hue="Loan_Approved", bins=20, multiple="dodge")

# leons get approved if the credit score tends to be above 650
# remove applicant Id
df =df.drop("Applicant_ID", axis=1)

df.head()
df.columns
df.info()

# Encoding: so we have binary encoding for maps and one hot encoding which is for get_dummies
# LabelEncoder: assigns an integer to each category
# OneHotEncoder: creates binary columns for each category
#F = 0 and M = 1
from sklearn.preprocessing import LabelEncoder # for ordinal data where there is order in the data
from sklearn.preprocessing import OneHotEncoder # for nominal data where there is no order in the data

# employment status for one hot encoding so that 1 = employed and 0 = unemployed
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index) 
df = pd.concat([df.drop(columns=cols), encoded_df],axis=1)

encoded

encoded_df.head()

df.head()

df.describe()

df.info()

df.describe()

df.info()

# correlation heatmap
# it is a visual representation of the relationships between numerical variables in a data set
# it shows correlation coefficent between two numeric variables 
# ranges from -1 to 1
# 1 = perfect positive correlation
# -1 = perfect negative correlation
# 0 = no linear correlation  
num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr()
num_cols.corr()["Loan_Approved"].sort_values(ascending=False)

plt.figure(figsize=(15,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
# we get quick insights and can detect multicollinearity (solved using feature, selection, PCA or dimensionality reduction)
# helps us with data exploration and preprocessing
