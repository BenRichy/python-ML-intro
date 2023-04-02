#my working file for the linear regression lesson

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from siuba import * 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv("USA_Housing.csv")

#quick info on dataset
data.head()
data.info()
data.describe()
data.columns

#quick plot
sns.pairplot(data)
#find the density of prices (what we want to model)
sns.distplot(data['Price'])
#correlation plot
data.corr()
#corr plot as heatmap
sns.heatmap(data.corr(), annot=True)

#create features
HouseFeatures = (data >>
select('Avg. Area Income', 
       'Avg. Area House Age', 
       'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 
       'Area Population'))

#target variable, needs to be a vector
HousePrices = data['Price']

#split data into training and test data
HouseFeatures_train, HouseFeatures_test, HousePrices_train, HousePrices_test = train_test_split(HouseFeatures,
                                                                                                HousePrices,
                                                                                                test_size = 0.4,
                                                                                                random_state=101)

#create linear reg model
lm = LinearRegression()
lm.fit(HouseFeatures_train, HousePrices_train)

#print intercepts
print(lm.intercept_)
#print coeffecients
print(lm.coef_)

#create coefficient data frame
cdf = pd.DataFrame(lm.coef_, HouseFeatures.columns, columns=['Coeff'])

#cdf shows the increase in house price per unit increase in the variable


