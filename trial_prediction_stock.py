import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# get various stock data
stock_name = yf.Ticker('CANF')
stock_data = stock_name.history(start=datetime.now() - timedelta(days=8000), end=datetime.now())
print("complete number of entries of the stock: ", len(stock_data))
print(stock_data.columns)
print(stock_data['Close'].head())
# # plotting the graph of the stock
# plt.figure(figsize=(16,8))
# stock_data['Close'].plot(title='CANF closing rate')
# plt.show()

# reading the given dataset
# df_stock = pd.read_csv('/home/richie/TAMU_hackathon/dataset/mystery_stock_daily_train.csv')
# print(len(df_stock))
# print("************************************")
# print(df_stock.columns)

# taking only the required column
df_stock_close = stock_data[['Close']]
future_days = 25
df_stock_close['Prediction'] = df_stock_close[['Close']].shift(-future_days)
X = np.array(df_stock_close.drop(['Prediction'], 1))[:-future_days]
y = np.array(df_stock_close['Prediction'])[:-future_days]
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
tree = DecisionTreeRegressor().fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)
X_future = df_stock_close.drop(['Prediction'], 1)[:-future_days]
X_future = X_future.tail(future_days)
X_future = np.array(X_future)
# predicting
tree_prediction = tree.predict(X_future)
lr_prediction = lr.predict(X_future)
print(tree.score(X_test, y_test))
print("lr: ", lr.score(X_test, y_test))
# save the model
file_name = 'linear_regression.sav'
pickle.dump(lr, open(file_name, 'wb'))