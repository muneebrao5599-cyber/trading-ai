from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

# Model train function (pehli baar run pe)
def train_model():
    # Dummy data for testing (real mein CSV se replace kar)
    data = pd.DataFrame({'close': np.random.rand(100) * 100})  # Fake prices
    prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Yeh line sahi indent pe hai

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)  # Testing ke liye kam epochs
    model.save('model.h5')
    return model, scaler

# Load model ya train if not exist
try:
    model = load_model('model.h5')
    scaler = MinMaxScaler()  # Real mein scaler bhi save karna better
except:
    model, scaler = train_model()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    # Dummy last price (real mein last 60 prices store karna padega)
    latest_prices = [data.get('close', 0)]
    input_data = np.array(latest_prices * 60).reshape(-1, 1)  # Repeat for dummy
    scaled_input = scaler.transform(input_data)[-60:].reshape(1, 60, 1)
    pred = scaler.inverse_transform(model.predict(scaled_input))[0][0]
    current = latest_prices[-1]
    if pred > current * 1.005:
        signal = "Buy"
    elif pred < current * 0.995:
        signal = "Sell"
    else:
        signal = "Hold"
    print(f"Signal: {signal} for price {current}")  # Console pe dikhega
    return jsonify({"signal": signal}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)