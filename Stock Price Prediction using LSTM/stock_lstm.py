import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
# Step 2: Load stock data using yfinance

# Apple company ka stock example (change kar sakti ho)
df = yf.download('AAPL', start='2015-01-01', end='2022-01-01')

# Top 5 rows print karke check kar lo
print(df.head())

data = df['Close'].values.reshape(-1, 1)  # Use only 'Close' price
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
sequence_len = 60
X = []
y = []

for i in range(sequence_len, len(scaled_data)):
    X.append(scaled_data[i-sequence_len:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 5: Reshape input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(real_prices, label='Real Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig("stock_prediction_output.png") 
plt.show()
