import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#1. Load the Dataset
data = pd.read_csv("/Users/rahulkumar/Downloads/Banking_Dataset.csv")

#2. Print first 10 rows
print (df.head(10))

# STEP 3: BASIC DATA CHECK
print("SHAPE:\n", df.shape)
print("\nINFO:\n")
data = df.info()
print("\nDESCRIBE:\n")
print(df.describe(include='all'))
print("\nCOLUMNS:\n")
print(df.columns)

# STEP 4: CHECK MISSING VALUES
df.isnull().sum() # null values
df.notnull().sum() # not-null values

# STEP 5: HEATMAP OF NULL VALUES
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True)
plt.title("Null Values Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.notnull(), cbar=True)
plt.title("Not-Null Values Heatmap")
plt.show()

# STEP 6: CHECK DUPLICATES
df.duplicated().sum()

# Drop duplicates
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# STEP 7: CHECK & FIX DATA TYPES
df.dtypes

# STEP 8: UNIVARIATE VISUALIZATIONS
# Histogram
plt.figure(figsize=(10,6))
plt.hist(df['age'])
plt.title("Histogram of Age")
plt.show()

# Countplot
plt.figure()
sns.countplot(x=df['marital'])  
plt.title("Countplot of marital")
plt.show()

# Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age")
plt.show()

# HISTOGRAM
plt.figure(figsize=(12, 6))
sns.countplot(x=df['education'])
plt.title('Countplot of Education')
plt.xlabel('education')
plt.ylabel('Count')
plt.show()

# STEP 9: BIVARIATE VISUALIZATIONS
# SCATTER PLOT
plt.figure(figsize=(10,6))
plt.scatter(df['duration'], df['campaign'], alpha=0.5)

plt.title("Scatter Plot: Duration vs Campaign")
plt.xlabel("duration")
plt.ylabel("campaign")

plt.show()

# LINE PLOT
plt.figure(figsize=(8,5))
sns.lineplot(x='emp_var_rate', y='cons_conf_idx', data=df)
plt.title('Employment Variation Rate vs Consumer Confidence Index')
plt.xlabel('emp_var_rate')
plt.ylabel('cons_conf_idx')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='y', y='duration', data=df)
plt.title("Barplot: Subscription (y) vs Call Duration")
plt.xlabel("y (Subscribed?)")
plt.ylabel("duration")
plt.show()


# ALL COLUMN NAMES
df.columns

# FEATURE SELECTION AND TARGET COLUMN
x = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
       'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']]
y = df['y']

# Convert Categorical Columns to Numeric
x = pd.get_dummies(x, drop_first=True)

#Train–Test Split
x_train, x_test, y_train, y_test= train_test_split(
x, y, test_size=0.2, random_state=42)

#Scaling (StandardScaler)
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr= lr.predict(x_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# KNN (K-Nearest Neighbors)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# SVC (Support Vector Classifier)
svc = SVC()
svc.fit(x_train, y_train)
y_pred_svc= svc.predict(x_test)
print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc))


from sklearn.metrics import accuracy_score
LR_accuracy = accuracy_score(y_test, y_pred_lr)
KNN_accuracy = accuracy_score(y_test, y_pred_knn)
SVC_accuracy = accuracy_score(y_test, y_pred_svc)

print("MODEL ACCURACY SUMMARY")
print("Logistic Regression Accuracy:", LR_accuracy)
print("KNN Accuracy:", KNN_accuracy)
print("SVC Accuracy:", SVC_accuracy)



# STEP 10: BAR GRAPH FOR MODEL ACCURACY COMPARISON

import matplotlib.pyplot as plt

models = ['Logistic Regression', 'KNN', 'SVC']
accuracies = [1.0, 0.97538778533236857, 0.9995143273433705]

colors = ['blue', 'orange', 'brown']

plt.figure(figsize=(10,6))
plt.bar(models, accuracies, color=colors)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of ML Models")
plt.ylim(0.8, 1.05) 
plt.xticks(rotation=15)

plt.show()
