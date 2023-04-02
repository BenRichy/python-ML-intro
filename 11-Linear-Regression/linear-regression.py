#my working file for the linear regression lesson

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from siuba import * 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



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


#now predict the model
HousePredictions = lm.predict(HouseFeatures_test)

#can plot a scatter to show how close we are
plt.scatter(HousePrices_test, HousePredictions)
#histogram of the residuals
sns.distplot((HousePrices_test-HousePredictions))
#model appears to be normally distributed, which is reasonable


#3 common evaluation metrics we can use are:
#Mean absolute error (MAE) - average error
#Mean Squared error (MSE) - 'punishes large errors', hence more useful
#Root mean squared error (RMSE) - makes the error interpretable in the same units as y

#following importing metrics from sklearn, calc the absolute error
metrics.mean_absolute_error(HousePrices_test, HousePredictions)
metrics.mean_squared_error(HousePrices_test, HousePredictions)
np.sqrt(metrics.mean_absolute_error(HousePrices_test, HousePredictions))




