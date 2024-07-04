import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Define the trading models
def moving_average_crossover(symbol, start_date, end_date, short_window=40, long_window=100, look_back_period=5):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    data['Signal'] = 0
    data.loc[data.index[short_window:], 'Signal'] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)

    # Adjust the Signal column based on lookback period
    for i in range(look_back_period, len(data)):
        recent_signals = data['Signal'].iloc[i-look_back_period+1:i+1]
        if recent_signals.iloc[-1] == 1 and 1 in recent_signals.values:
            data.at[data.index[i], 'Signal'] = 1
        elif recent_signals.iloc[-1] == 0:
            data.at[data.index[i], 'Signal'] = 0
        else:
            data.at[data.index[i], 'Signal'] = -1

    return data

def bollinger_bands(symbol, start_date, end_date, window=20, num_std_dev=2):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Rolling_Mean'] = data['Close'].rolling(window=window, min_periods=1).mean()
    data['Bollinger_Up'] = data['Rolling_Mean'] + num_std_dev * data['Close'].rolling(window=window, min_periods=1).std()
    data['Bollinger_Down'] = data['Rolling_Mean'] - num_std_dev * data['Close'].rolling(window=window, min_periods=1).std()
    data['Signal'] = 0
    data.loc[data['Close'] < data['Bollinger_Down'], 'Signal'] = 1
    data.loc[data['Close'] > data['Bollinger_Up'], 'Signal'] = -1
    return data

def relative_strength_index(symbol, start_date, end_date, window=14):
    data = yf.download(symbol, start=start_date, end=end_date)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Signal'] = 0
    data.loc[data['RSI'] < 30, 'Signal'] = 1
    data.loc[data['RSI'] > 70, 'Signal'] = -1
    return data

def sma_mean_reversion(symbol, start_date, end_date, window=20, threshold=0.05):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['SMA'] = data['Close'].rolling(window=window, min_periods=1).mean()
    data['Threshold_Up'] = data['SMA'] * (1 + threshold)
    data['Threshold_Down'] = data['SMA'] * (1 - threshold)
    data['Signal'] = 0
    data.loc[data['Close'] < data['Threshold_Down'], 'Signal'] = 1
    data.loc[data['Close'] > data['Threshold_Up'], 'Signal'] = -1
    return data

# Load the pre-trained model
model_2 = joblib.load('model_2.pkl')

def random_forest_predictions(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.drop(columns=['Adj Close'], inplace=True, errors='ignore')
    # Ensure the data has the necessary features the model was trained on
    feature_columns = ['Close', 'Volume', 'Open', 'High', 'Low']
    if not all(col in data.columns for col in feature_columns):
        raise ValueError("Input data does not have the required features")
    # Reorder columns to match the order during model training
    data = data[feature_columns]
    preds = model_2.predict(data)
    return preds, data

# Function to combine the outputs of multiple trading models above into a dataframe
def combine_model_outputs(symbol, start_date, end_date, fees=0.01):
    # Apply trading models
    ma_crossover_data = moving_average_crossover(symbol, start_date, end_date)
    bb_data = bollinger_bands(symbol, start_date, end_date)
    rsi_data = relative_strength_index(symbol, start_date, end_date)
    sma_mr_data = sma_mean_reversion(symbol, start_date, end_date)

    # Combine DataFrames
    combined_data = ma_crossover_data[['Close', 'Volume', 'Short_MA', 'Long_MA']].copy()
    combined_data['MA_Signal'] = ma_crossover_data['Signal']
    combined_data['BB_Signal'] = bb_data['Signal']
    combined_data['RSI_Signal'] = rsi_data['Signal']
    combined_data['SMA_Signal'] = sma_mr_data['Signal']

    # Get Random Forest predictions
    preds, feature_data = random_forest_predictions(symbol, start_date, end_date)
    combined_data['RF_Prediction'] = preds

    # Calculate the target variable
    combined_data['Future_Return'] = combined_data['Close'].pct_change().shift(-1)
    combined_data['Target'] = combined_data['Future_Return'].apply(lambda x: 1 if x > fees else (-1 if x < -fees else 0))

    # Ensure the target variable is NaN for the most recent date
    combined_data.at[combined_data.index[-1], 'Future_Return'] = np.nan
    combined_data.at[combined_data.index[-1], 'Target'] = np.nan

    return combined_data


combine_model_outputs('AAPL', '2021-01-01', '2021-12-31')

