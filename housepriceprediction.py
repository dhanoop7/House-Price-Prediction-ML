#importing the dependencies vvvvvvv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


house_price_dataset = sklearn.datasets.fetch_california_housing() #importing the boston house price dataset

#DATA ANALYSING STEPS vvvvvvvvvvv

house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names) # loading the dataset to a Pandas dataframe

house_price_dataframe['price'] = house_price_dataset.target #adding the target (price) column to the dataframe

print("#printing the rows")
print(house_price_dataframe.head()) #printing the rows

print("#checking the number of rows and column in the dataframe")
print(house_price_dataframe.shape) #checking the number of rows and column in the dataframe

print("# check for missing values")
print(house_price_dataframe.isnull().sum()) # check for missing values

print("statistical measures of dataset")
print(house_price_dataframe.describe()) # statistical measures of dataset

# understanding the correlation btw various features in the dataset they can be 1. possitive 2. neagative (DATA ANALYSING) vvvvvvv

correlation = house_price_dataframe.corr()


print("# constructing a heatmap to understand the correlation")
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues') # constructing a heatmap to understand the correlation
plt.show()

# spliting the data and target
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

print("price and data")
print(X)
print(Y)

#spliting the data into Trainig data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 2)

print("Seeing how many instances are there")
print(X.shape, X_train.shape, X_test.shape)

#MODEL TRAINING (XGBoost Regressor)

model = XGBRegressor() #loading the model

model.fit(X_train, Y_train) #traing the model

#accuracy for prediction on training data
trainig_data_prediction = model.predict(X_train)

print(trainig_data_prediction)

score_1 = metrics.r2_score(Y_train, trainig_data_prediction) #R squared error

score_2 = metrics.mean_absolute_error(Y_train, trainig_data_prediction) # mean absolute error

plt.scatter(Y_train, trainig_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()


print("R squared error: ", score_1)
print("Mean absolute Error: ", score_2)

#prediction on test data
test_data_prediction = model.predict(X_test)

score_1 = metrics.r2_score(Y_test, test_data_prediction) #R squared error

score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction) # mean absolute error

print("R squared error: ", score_1)
print("Mean absolute Error: ", score_2)


# Collecting new data from the user
new_user_data = {}

new_user_data['MedInc'] = float(input("Enter MedInc: "))
new_user_data['HouseAge'] = int(input("Enter HouseAge: "))
new_user_data['AveRooms'] = float(input("Enter AveRooms: "))
new_user_data['AveBedrms'] = float(input("Enter AveBedrms: "))
new_user_data['Population'] = float(input("Enter Population: "))
new_user_data['AveOccup'] = float(input("Enter AveOccup: "))
new_user_data['Latitude'] = float(input("Enter Latitude: "))
new_user_data['Longitude'] = float(input("Enter Longitude: "))

print("New data:", new_user_data)

new_user_data_dataframe = pd.DataFrame([new_user_data])

new_data_prediction = model.predict(new_user_data_dataframe)

print("The predicted Price is : ", new_data_prediction)
