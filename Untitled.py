#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# # Data Collection & Analysis

# In[7]:


# loading the data from csv file to a pandas Dataframe
insurance_dataset = pd.read_csv("C:/Users/acer/Medical Insurance Cost Predicter/insurance.csv")


# In[9]:



insurance_dataset.head()


# In[10]:


# number of rows and columns
insurance_dataset.shape


# In[11]:


#getting some information about the dataset
insurance_dataset.info()

# categorical features:
sex
smoke
region
# In[13]:


#checking for missing values
insurance_dataset.isnull().sum()


# # Data Analysis

# In[14]:


#statistical measures of the dataset
insurance_dataset.describe()


# In[19]:


# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()


# In[21]:


# Gender column
sns.set()
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[22]:


insurance_dataset['sex'].value_counts()


# In[23]:


# distribution of bmi
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[ ]:


#normal BMI Range -> 18.5 to 24.9


# In[24]:


# count column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()


# In[25]:


insurance_dataset['children'].value_counts()


# In[26]:


# distribution of smoker column
# sns.set()
plt.figure(figsize=(6,6))
sns.countplot(insurance_dataset['smoker'])
plt.title('Smoker')
plt.show()


# In[28]:


insurance_dataset['smoker'].value_counts()


# In[30]:


# distribution of region column
# sns.set()
plt.figure(figsize=(6,6))
sns.countplot(insurance_dataset['region'])
plt.title('Region')
plt.show()


# In[31]:


# distribution of charges column
# sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# # Data Pre-Processing

# In[ ]:


#Encoding the categorical features


# In[32]:


#Encoding sex column
insurance_dataset.replace({'sex':{'male':0, 'female':1}}, inplace=True)


# In[35]:


#encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)

#Encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)


# # Splitting the Features and Target

# In[36]:


X= insurance_dataset.drop(columns='charges', axis=1)
Y= insurance_dataset['charges']


# In[37]:


print(X)


# In[38]:


print(Y)


# # Splitting the data into Training data & Testing Data

# In[39]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[40]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training
# 
# #Linear Regression

# In[41]:


#loading the Linear Regression Model
regressor = LinearRegression()


# In[42]:


regressor.fit(X_train, Y_train)


# # Model Evaluation

# In[43]:


# prediction on training data
training_data_prediction = regressor.predict(X_train)


# In[44]:


# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value : ', r2_train)


# In[45]:


# prediction on test data
test_data_prediction = regressor.predict(X_test)


# In[47]:


# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value : ', r2_test)


# # Build a Predictive System

# In[49]:


# input_data = (31,female,25.74,0,no,southeast)
input_data = (31,1,25.74,0,1,0)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)


# In[50]:


# input_data = (31,female,25.74,0,no,southeast)
input_data = (22,0,23.74,0,1,0)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

