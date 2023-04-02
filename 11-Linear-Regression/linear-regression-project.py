#this script is my pure python version of the jupiter notebook

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from siuba import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#read in Ecommerce dataset
customers = pd.read_csv("Ecommerce Customers")

#create jointplot to compare time on website and yearly spend
sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')
#data is normally distributed

#create 2D hex bin plot of time on app vs length of membership
sns.jointplot(data=customers, x='Time on App', y='Length of Membership', kind='hex')

#create pairplot to compare all relationships
sns.pairplot(data=customers)
#length of membership and yearly amount spent are the most correlated data

#create linear model plot of yearly amount spent vs length of membership
sns.lmplot(data=customers,x='Yearly Amount Spent', y='Length of Membership')

#split data into training and test data
#select only numeric columns for the features (x dataset)
CommerceFeatures = customers.select_dtypes(include=np.number).drop(columns=['Yearly Amount Spent'])
#select yearly amount spent column as the target variable
CommerceTarget = customers['Yearly Amount Spent']

#split data into training and test data
CommerceFeatures_train, CommerceFeatures_test, CommerceTarget_train, CommerceTarget_test = train_test_split(CommerceFeatures, CommerceTarget, test_size = 0.3, random_state=101)

#train the model

#create LinearRegression instance
lm = LinearRegression()
lm.fit(CommerceFeatures_train, CommerceTarget_train)

#print coeffecients
print(lm.coef_)

#predict test data
CommercePredict = lm.predict(CommerceFeatures_test)

#scatter plot to show real test vs predicted values
plt.scatter(CommerceTarget_test, CommercePredict)

#evaluate the model by obtainting MAE, MSE, RMSE
print(metrics.mean_absolute_error(CommerceFeatures_test, CommercePredict))

print(metrics.mean_squared_error(CommerceFeatures_test, CommercePredict))
print(np.sqrt(metrics.mean_absolute_error(CommerceFeatures_test, CommercePredict)))
