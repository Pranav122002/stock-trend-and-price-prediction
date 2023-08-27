# Stock Trend and Price Prediction
This project gives Buy and Sell signals based on predicted Trend by Simple Moving Averages and  predicts stock prices for a company using it's historical stock price data using LSTM model ( Long Short Term Memory ).

* Data is fetched from yahoo finance API.
* Closing Price is set as dependent variable.
* Simple Moving Average Strategy is used to predict Buy and Sell Signals.
* LSTM Model is used to predict stock prices.

## Demo Link
See live demo at https://pranav122002-stock-trend-and-price-prediction.streamlit.app/

## Installation
```
git clone https://github.com/Pranav122002/stock-trend-and-price-prediction.git
```
```
cd .\stock-trend-and-price-prediction\
```
## Install required modules and packages
```
pip install numpy pandas streamlit scikit-learn matplotlib yfinance keras 
```
or
```
pip install -r requirements.txt
```

## Running app
```
streamlit run app.py
```

You can get Stock Tickers from yahoo finance website by searching company's name at https://finance.yahoo.com/
