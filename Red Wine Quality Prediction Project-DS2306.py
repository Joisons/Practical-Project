#!/usr/bin/env python
# coding: utf-8

# ## Red Wine Quality Prediction Project Project

# **Prepared for FLIP ROBO by Olumide Ikumapayi-DS2306**

# ## About the Wine Dataset

# The dataset provided in this project focuses on the classification of red and white varieties of Portuguese "Vinho Verde" wine. Due to privacy and logistical limitations, only physicochemical attributes (inputs) and sensory evaluations (the output) are available for analysis, while specific information regarding grape varieties, wine brands, and selling prices has been excluded. As a result, this dataset can be utilized for classification purposes.For more information you can check the link to the dataset https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv 

# ## Import Necessary Libraries

# In[60]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve


# ## Importing and Preprocessing the Dataset

# In[61]:


# URL to the raw CSV file on GitHub
url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv"

# Reading the CSV file into a DataFrame
wine_quality = pd.read_csv(url)


# In[62]:


df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv") 
df


# In[63]:


df.head(15)


# In[64]:


df.tail(30)


# # Exploratory Data Analysis(EDA)

# In[65]:


#Checking the dimension of the dataset
df.shape


# This dataset contains 1599 rows and 12 columns.Out of which 1 is target variable and the remaining 11 are independent variables.

# In[66]:


df.columns


# In[67]:


# Checking the columns of dataset
df.columns.tolist()


# In[68]:


# checking the types of columns
df.dtypes


# There are two(2) types of data(int64 and float) present in the dataset
# 
# **Attribute Information**
# 
# Input variables (based on physicochemical tests):
# * fixed acidity
# * volatile acidity
# * citric acid
# * residual sugar
# * chlorides
# * free sulfur dioxide
# * total sulfur dioxide
# * density
# * pH
# * sulphates
# * alcohol
# * Output variable (based on sensory data):
# * quality (score between 0 and 10)
# 

# In[69]:


#checking the null values
df.isnull().sum()


# In[70]:


df.isnull().sum().sum()


# As seen above,there were no null values present in this dataset.

# In[71]:


# Separating numerical and categorical columns

#checking for categorical columns
categorical_col = []
for i in df.dtypes.index:
    if df.dtypes[i] == "object":
        categorical_col.append(i)
print("categorical columns:", categorical_col)
print("\n")

#Checking for Numerical column
numerical_col = []
for i in df.dtypes.index:
    if df.dtypes[i]!= "object":
        numerical_col.append(i)
print("numerical columns:", numerical_col)


# In[72]:


# Checking number of unique values in each column
df.nunique().to_frame("No.of unique values")


# In[73]:


df.head()


# In[74]:


# Checking the list of counts of target
df["quality"].unique()


# In[75]:


# Checking the unique values in target column
df['quality'].value_counts()


# In[76]:


#Lets visualize using heatmap
sns.heatmap(df.isnull())


# As clearly visualize there is no missing data present.

# In[77]:


# To get good overview of the dataset
df.info()


# This provides a concise overview of the dataset, including information on the indexing type, column type, absence of null values, and memory usage.

# In[78]:


# Checking the value counts of each column
for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# These are the value counts of all columns

# In[79]:


# Lets check the null values again
sns.heatmap(df.isnull(),cmap = "cool_r")


# ## Data Preprocessing

# In[80]:


# Setting the cutoff for 'good' wines
cutoff = 7


# In[81]:


# Creating a binary classification target variable
wine_quality['is_good'] = (wine_quality['quality'] >= cutoff).astype(int)


# In[82]:


# Drop the original 'quality' column
wine_quality.drop('quality', axis=1, inplace=True)


# In[83]:


# Split the dataset into features (X) and target (y)
X = wine_quality.drop('is_good', axis=1)
y = wine_quality['is_good']


# In[84]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[85]:


# Standardizing the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Decision Tree Classifier:

# In[86]:


# Creating a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Fitting the classifier to the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)


# ## Model Evaluation and plot the ROC curve

# In[89]:


# printing accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# printing precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')

# printing recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.2f}')

# printing F1-score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.2f}')

# Printing confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Printing the classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)


# In[90]:


# Calculating the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ## Summary and Conclusion

# The Decision Tree classification model exhibits favorable performance for Class 0, demonstrating a precision of 0.92 and recall of 0.93, resulting in an F1-score of 0.93. In contrast, Class 1 displays lower precision at 0.57 and recall at 0.51, yielding an F1-score of 0.54. The overall accuracy stands at 0.87 while the ROC-AUC score is recorded as 0.72, suggesting satisfactory model performance with potential for enhancement, particularly concerning Class 1 predictions.

# In[ ]:




