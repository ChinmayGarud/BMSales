# Importing basic libraries
import numpy as np
import pandas as pd


"""**Importing the datasets**"""

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


train.head()

train.describe()
test.describe()



"""  Data PreProcessing  """


# Handling missing values

train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace = True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace = True)

train.isnull().sum()

# Combining reg, Regular and Low Fat, low fat and, LF

train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
train['Item_Fat_Content'].value_counts()


# Determining the operation peroid of store

train['Outlet_Years'] = 2013 - train['Outlet_Establishment_Year']
train['Outlet_Years'].value_counts()

# Removing unnecassary columns from the dataset
train = train.drop('Item_Identifier', axis = 1)
train = train.drop('Outlet_Identifier', axis = 1)
train = train.drop('Outlet_Establishment_Year', axis = 1)
print(train.shape)

train['Outlet_Type'].value_counts()

# OneHot Encoding
train = pd.get_dummies(train)

print(train.shape)

# Splitting the data into dependent and target variables

X = train.drop('Item_Outlet_Sales', axis = 1)
y = train.Item_Outlet_Sales


# splitting the dataset into X_train, X_test, y_train, y_test
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


"""
  MODEL BUILDING"""

"""Linear Regression"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the  test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# finding the mean squared error and variance
mse = mean_squared_error(y_test, y_pred)
print('RMSE :', np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


"""Gradient Boosting Regressor"""
from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# Calculating the root mean squared error
print("RMSE :", np.sqrt(((y_test - y_pred)**2).sum()/len(y_test)))

""" Random Forest Regression """
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100 , n_jobs = -1)
regressor.fit(X_train, y_train)

# predicting the  test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# finding the mean squared error and variance
mse = mean_squared_error(y_test, y_pred)
print("RMSE :",np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",regressor.score(X_train, y_train))

"""  Decision Tree Regressor  """
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)
print(y_pred)

print(" RMSE : " , np.sqrt(((y_test - y_pred)**2).sum()/len(y_test)))

'''
    USING RMSE AS AN EVALUATION METRIC FOR MODEL
    GRADIENT BOOSTING IS THE BEST REGRESSOR HERE
'''

""" TEST DATASET """

# Preprocessing of test data

# Handling missing values
test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace = True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace = True)
test.isnull().sum()

# Combining reg, Regular and Low Fat, low fat and, LF
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
test['Item_Fat_Content'].value_counts()

# Determining the operation peroid of store
test['Outlet_Years'] = 2013 - test['Outlet_Establishment_Year']
test['Outlet_Years'].value_counts()

# Removing unnecassary columns from the dataset
test = test.drop('Item_Identifier', axis = 1)
test = test.drop('Outlet_Identifier', axis = 1)
test = test.drop('Outlet_Establishment_Year', axis = 1)
print(test.shape)

test['Outlet_Type'].value_counts()

# OneHot Encoding
test = pd.get_dummies(test)

print(test.shape)


# Predicting on test data

pred_test = regressor.predict(test)

print(pred_test)
