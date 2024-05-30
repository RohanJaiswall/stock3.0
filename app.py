import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler 
import streamlit as st
import matplotlib.pyplot as plt


import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('C:/Users/rohan/stock  market/Stock_Predictions_Model1.keras')

# Set up the Streamlit app layout
st.title('Stock Market Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start_date = '2012-01-01'
end_date = '2022-12-31'

# Download stock data
data = yf.download(stock, start=start_date, end=end_date)

# Display stock data in a Streamlit table
st.subheader('Stock Data')
st.write(data)

# Preprocess data for prediction
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plotting functions
def plot_price_ma(data, ma_days, title):
    ma = data.Close.rolling(ma_days).mean()
    plt.figure(figsize=(8, 6))
    plt.plot(ma, 'r', label=f'MA {ma_days}')
    plt.plot(data.Close, 'g', label='Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    return plt

# Plot price vs MA50
st.subheader('Price vs MA50')
fig1 = plot_price_ma(data, 50, 'Price vs MA50')
st.pyplot(fig1)

# Plot price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
fig2 = plot_price_ma(data, 50, 'Price vs MA50 vs MA100')
plot_price_ma(data, 100, 'Price vs MA50 vs MA100')
st.pyplot(fig2)

# Plot price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
fig3 = plot_price_ma(data, 100, 'Price vs MA100 vs MA200')
plot_price_ma(data, 200, 'Price vs MA100 vs MA200')
st.pyplot(fig3)

# Prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

# Plot original price vs predicted price
st.subheader('Original Price vs Predicted Price')
plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot()

# Optionally, you can also display the individual predicted and original prices
st.subheader('Predicted Prices')
st.write(predict)

st.subheader('Original Prices')
st.write(y)
