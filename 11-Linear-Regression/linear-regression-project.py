#this script is my pure python version of the jupiter notebook

#import packages
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns 
from siuba import *

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