#!/usr/bin/env python
# coding: utf-8

# ## Medical Cost Personal Insurance Project

# **Prepared for FLIP ROBO by Olumide Ikumapayi-DS2306**

# ## About the Medical Cost Personal Dataset

# The dataset relates to the Health insurance which is a form of insurance that provides coverage for medical expenses resulting from an illness. These expenses may include costs associated with hospitalization, medication, and consultations with doctors. The primary goal of medical insurance is to ensure access to high-quality healthcare without incurring financial burden. Health insurance plans offer protection against exorbitant medical expenses, including those related to hospital stays, outpatient procedures, at-home care, and transportation by ambulance. The calculation of medical insurance premiums takes into account various factors such as age, body mass index (BMI), number of dependents, smoking status, and geographical region.For more information you can check the link to the dataset https://github.com/dsrscientist/dataset4/blob/main/medical_cost_insurance.csv

# ## Import Necessary Libraries

# In[42]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ## Importing and Preprocessing the Dataset

# In[44]:


# URL to the raw CSV file on GitHub
url = "https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv"

# Reading the CSV file into a DataFrame
medical_cost_insurance = pd.read_csv(url)


# In[46]:


df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv") 
df


# In[48]:


df.head(15)


# In[60]:


df.tail(30)


# # Exploratory Data Analysis(EDA)

# In[61]:


#Checking the dimension of the dataset
df.shape


# This dataset contains 1338 rows and 7 columns.Out of which 1 is target variable and the remaining 6 are independent variables.

# In[62]:


df.columns


# In[63]:


# Checking the columns of dataset
df.columns.tolist()


# In[64]:


# checking the types of columns
df.dtypes


# There are three(3) types of data(int64,object and float) present in the dataset.
# 
# 
# **The Columns** 
# * **age**: This are the age of primary beneficiary
# * **sex**: This are the insurance contractor gender, female, male
# * **bmi**: Meaning Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9.
# * **children**: This are the Number of children covered by health insurance / Number of dependents
# * **smoker**: Smoking
# * **region**: This are the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# * **charges**: Are the Individual medical costs billed by health insurance
# 

# In[65]:


#checking the null values
df.isnull().sum()


# In[66]:


df.isnull().sum().sum()


# As seen above,there were no null values present in this dataset.

# In[67]:


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


# In[68]:


# Checking number of unique values in each column
df.nunique().to_frame("No.of unique values")


# In[69]:


df.head()


# In[70]:


# Checking the list of counts of target
df["charges"].unique()


# In[71]:


# Checking the unique values in target column
df['charges'].value_counts()


# In[72]:


#Lets visualize using heatmap
sns.heatmap(df.isnull())


# As clearly visualize there is no missing data present.

# In[73]:


# To get good overview of the dataset
df.info()


# This provides a concise overview of the dataset, including information on the indexing type, column type, absence of null values, and memory usage.

# In[74]:


# Checking the value counts of each column
for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# These are the value counts of all columns

# In[75]:


# Lets check the null values again
sns.heatmap(df.isnull(),cmap = "cool_r")


# ## Data Preprocessing

# In[76]:


# Encoding categorical features for the training data
label_encoder = LabelEncoder()
medical_cost_insurance['sex'] = label_encoder.fit_transform(medical_cost_insurance['sex'])
medical_cost_insurance['smoker'] = label_encoder.fit_transform(medical_cost_insurance['smoker'])
medical_cost_insurance['region'] = label_encoder.fit_transform(medical_cost_insurance['region'])


# In[77]:


# Split the data into features (X) and the target variable (y):
X = medical_cost_insurance.drop(['charges'], axis=1)
y = medical_cost_insurance['charges']


# In[78]:


# Splitting the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Machine Learning Model

# ## Linear Regression Model

# In[79]:


# Creating a linear regression model
model = LinearRegression()


# In[80]:


# Training the model on the training data
model.fit(X_train, y_train)


# In[81]:


# Making predictions on the test data
y_pred = model.predict(X_test)


# ## Evaluating the Model

# In[82]:


# Evaluating the Linear Regression Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# ## Making Predictions for the LR Model

# In[83]:


# Creating a new DataFrame for the new individual
new_medical_cost_insurance = pd.DataFrame({
    'age': [30],
    'sex': ['female'],
    'bmi': [25],
    'children': [2],
    'smoker': ['no'],
    'region': ['northeast']
})


# In[84]:


# Encoding categorical features for the new data using the same LabelEncoder instance
label_encoder = LabelEncoder()
new_medical_cost_insurance['sex'] = label_encoder.fit_transform(new_medical_cost_insurance['sex'])
new_medical_cost_insurance['smoker'] = label_encoder.fit_transform(new_medical_cost_insurance['smoker'])
new_medical_cost_insurance['region'] = label_encoder.fit_transform(new_medical_cost_insurance['region'])


# In[85]:


# prediction insurance cost using Linear Regression Model
predicted_cost = model.predict(new_medical_cost_insurance)
print(f"Predicted Insurance Cost: {predicted_cost[0]}")


# In[86]:


# Scatter plots to visualize by Ploting actual vs. predicted values on the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Linear Regression, Actual vs. Predicted Insurance Costs')
plt.xlabel('Actual Costs')
plt.ylabel('Predicted Costs')
plt.grid(True)

# Display the Mean Squared Error (MSE) on the plot
plt.annotate(f'MSE: {mse:.2f}', xy=(3000, 40000), fontsize=12, color='red')

# Plot a diagonal line for reference
plt.plot([0, np.max(y_test)], [0, np.max(y_test)], color='red', linestyle='--')

plt.show()


# ### Random Forest Regressor Model

# In[87]:


# Creating a Random Forest model
model = RandomForestRegressor()


# In[88]:


# Training the model on the training data
model.fit(X_train, y_train)


# In[89]:


# Making predictions on the test data
y_pred = model.predict(X_test)


# In[90]:


# Evaluating the Random Forest model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[91]:


# Creating a new DataFrame for the new individual
new_medical_cost_insurance = pd.DataFrame({
    'age': [30],
    'sex': ['female'],
    'bmi': [25],
    'children': [2],
    'smoker': ['no'],
    'region': ['northeast']
})


# In[92]:


# Encoding categorical features for the new data using the same LabelEncoder instance
label_encoder = LabelEncoder()
new_medical_cost_insurance['sex'] = label_encoder.fit_transform(new_medical_cost_insurance['sex'])
new_medical_cost_insurance['smoker'] = label_encoder.fit_transform(new_medical_cost_insurance['smoker'])
new_medical_cost_insurance['region'] = label_encoder.fit_transform(new_medical_cost_insurance['region'])


# In[93]:


# prediction insurance cost Using the Random Forest Model
predicted_cost = model.predict(new_medical_cost_insurance)
print(f"Predicted Insurance Cost: {predicted_cost[0]}")


# In[94]:


# Scatter plots to visualize by Ploting actual vs. predicted values on the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Random Foorest, Actual vs. Predicted Insurance Costs')
plt.xlabel('Actual Costs')
plt.ylabel('Predicted Costs')
plt.grid(True)

# Display the Mean Squared Error (MSE) on the plot
plt.annotate(f'MSE: {mse:.2f}', xy=(3000, 40000), fontsize=12, color='red')

# Plot a diagonal line for reference
plt.plot([0, np.max(y_test)], [0, np.max(y_test)], color='red', linestyle='--')

plt.show()


# ## DECISION TREE

# In[95]:


# Creating a Decision Tree model
model = DecisionTreeRegressor()


# In[96]:


# Training the model on the training data
model.fit(X_train, y_train)


# In[97]:


# Making predictions on the test data
y_pred = model.predict(X_test)


# In[98]:


# Evaluating the Decision Tree model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[99]:


# Creating a new DataFrame for the new individual
new_medical_cost_insurance = pd.DataFrame({
    'age': [30],
    'sex': ['female'],
    'bmi': [25],
    'children': [2],
    'smoker': ['no'],
    'region': ['northeast']
})


# In[100]:


# Encode categorical features for the new data using the same LabelEncoder instance
label_encoder = LabelEncoder()
new_medical_cost_insurance['sex'] = label_encoder.fit_transform(new_medical_cost_insurance['sex'])
new_medical_cost_insurance['smoker'] = label_encoder.fit_transform(new_medical_cost_insurance['smoker'])
new_medical_cost_insurance['region'] = label_encoder.fit_transform(new_medical_cost_insurance['region'])


# In[101]:


# prediction insurance cost Using Decision Tree Model
predicted_cost = model.predict(new_medical_cost_insurance)
print(f"Predicted Insurance Cost: {predicted_cost[0]}")


# In[102]:


# Scatter plots to visualize by Ploting actual vs. predicted values on the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Decision Tree, Actual vs. Predicted Insurance Costs')
plt.xlabel('Actual Costs')
plt.ylabel('Predicted Costs')
plt.grid(True)

# Display the Mean Squared Error (MSE) on the plot
plt.annotate(f'MSE: {mse:.2f}', xy=(3000, 40000), fontsize=12, color='red')

# Plot a diagonal line for reference
plt.plot([0, np.max(y_test)], [0, np.max(y_test)], color='red', linestyle='--')

plt.show()


# In[103]:


# Creating a K-Nearest Neighbors(KNN) model
model = KNeighborsRegressor()


# In[104]:


# Training the model on the training data
model.fit(X_train, y_train)


# In[105]:


# Making predictions on the test data
y_pred = model.predict(X_test)


# In[106]:


# Evaluating the K-Nearest Neighbor model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[107]:


# Creating a new DataFrame for the new individual
new_medical_cost_insurance = pd.DataFrame({
    'age': [30],
    'sex': ['female'],
    'bmi': [25],
    'children': [2],
    'smoker': ['no'],
    'region': ['northeast']
})


# In[108]:


# Encode categorical features for the new data using the same LabelEncoder instance
label_encoder = LabelEncoder()
new_medical_cost_insurance['sex'] = label_encoder.fit_transform(new_medical_cost_insurance['sex'])
new_medical_cost_insurance['smoker'] = label_encoder.fit_transform(new_medical_cost_insurance['smoker'])
new_medical_cost_insurance['region'] = label_encoder.fit_transform(new_medical_cost_insurance['region'])


# In[109]:


# predicting insurance cost using KNN
predicted_cost = model.predict(new_medical_cost_insurance)
print(f"Predicted Insurance Cost: {predicted_cost[0]}")


# In[110]:


# Scatter plots to visualize by Ploting actual vs. predicted values on the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('K-Nearest Neigbhor(KNN), Actual vs. Predicted Insurance Costs')
plt.xlabel('Actual Costs')
plt.ylabel('Predicted Costs')
plt.grid(True)

# Display the Mean Squared Error (MSE) on the plot
plt.annotate(f'MSE: {mse:.2f}', xy=(3000, 40000), fontsize=12, color='red')

# Plot a diagonal line for reference
plt.plot([0, np.max(y_test)], [0, np.max(y_test)], color='red', linestyle='--')

plt.show()


# ## Summary and Conclusion
This project assignment focused on predicting health insurance costs using a range of machine learning models. Factors such as age, BMI, dependents, smoking status, and region were considered in the estimation process. The results demonstrated significant variations in the predictions made by different models. For example, Linear Regression projected an insurance cost of $5,009.80. While Random Forest yielded a much higher estimate of $11,137.64. 

Decision Tree estimated the cost to be $4,435.09, and K-Nearest Neighbors (KNN) predicted a cost of $10,768.73. When selecting a model for this task, it is important to consider both performance metrics and interpretability.

These predictions provide valuable insights for insurers as they inform pricing strategies and enhance understanding of the factors that influence medical insurance expenses. Further evaluation and business applications can help refine decision-making processes within the insurance industry.

# In[ ]:




