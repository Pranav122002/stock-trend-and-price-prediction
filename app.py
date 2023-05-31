import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
import yfinance as yf

# Stock Tickers - GOOG, GAIL.NS, AAPL, HDFCBANK.NS 

# Importing Trained Model
model = load_model('stock_model.h5')

# Title
st.title('Stock Trend and Price Prediction')

# Settings
start = '2016-01-01'
end = '2019-12-31'
st.text("Example stock tickers are - GOOG, GAIL.NS, AAPL, HDFCBANK.NS ")
user_input = st.text_input("Enter Stock Ticker : ", 'GOOG')

# Making df
yf.pdr_override()
df = pdr.get_data_yahoo(user_input, start, end)
df = df.reset_index()
st.subheader('Data from 2018 - 2022')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig1 = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.xlabel('Time',fontsize=12)
plt.ylabel('Close Price',fontsize=12)
st.pyplot(fig1)

# Moving Average Function
def MA(data, period, column='Close'):
    return data[column].rolling(window=period).mean()

# Creating Moving Averages with different Days
df['MA20']=MA(df, 10)
df['MA50']=MA(df, 70)
df['MA100']=MA(df, 100)

# Creating Buy and Sell Signals
df['Signal']= np.where(df['MA20'] > df['MA50'], 1, 0)
df['Position'] = df['Signal'].diff()

df['Buy']= np.where(df['Position'] == 1, df['Close'], np.NAN)
df['Sell']= np.where(df['Position'] == -1, df['Close'], np.NAN)

# Plotting Moving Averages with Buy and Sell Signals
st.subheader('Moving Averages with Buy and Sell Signals')
fig2 = plt.figure(figsize = (12,6))
# plt.title('Close Price History with Buy and Sell Signals', fontsize=18)
plt.plot(df['Close'], alpha=0.5, label='Close')
plt.plot(df['MA20'], alpha=1, label='MA20', color='green')
plt.plot(df['MA50'], alpha=1, label='MA50', color='red')
plt.scatter(df.index, df['Buy'], alpha=1, label='Buy Signal', marker='^', color='green')
plt.scatter(df.index, df['Sell'], alpha=1, label='Sell Signal', marker='v', color='red')
plt.xlabel('Time',fontsize=12)
plt.ylabel('Close Price',fontsize=12)
plt.legend()
st.pyplot(fig2)

#Splitting into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =  pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Scaling down Data
scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)


#Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

#scaling down inputs
inputs = scaler.transform(final_df)

x_test = [] 
y_test = []

# x-test and y-test
for i in range(100, (inputs.shape[0])):
    x_test.append(inputs[i-100: i])
    y_test.append(inputs[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = (model.predict(x_test)) 

# Scaling up values
scale = scaler.scale_[0]

y_predicted = (1/scale) * y_predicted
y_test =  (1/scale) * y_test

# Plotting Predictons vs Actual Chart
st.subheader('Predictons vs Actual')
fig3 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'blue', label = 'Original Price')
plt.plot(y_predicted, 'orange', label = 'Predicted Price')
plt.xlabel('Time',fontsize=12)
plt.ylabel('Close Price',fontsize=12)
plt.legend()
st.pyplot(fig3)