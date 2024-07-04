import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import joblib
from trading_models import moving_average_crossover, bollinger_bands, relative_strength_index, sma_mean_reversion, combine_model_outputs, random_forest_predictions
import os
import pandas_datareader.data as pdr
from fredapi import Fred
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from pandas.tseries.offsets import DateOffset
import altair as alt


# Alpaca API credentials (replace with your own)
ALPACA_API_KEY = 'your_alpaca_api_key'
ALPACA_SECRET_KEY = 'your_alpaca_secret_key'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Set up Alpaca API client
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')


# Define functions for each page
@st.cache_data
def fetch_data(symbol, start, end):
    return yf.download(symbol.strip(), start, end)

# Load the pre-trained models
model = joblib.load('trained_model.pkl')
model_2 = joblib.load('model_2.pkl')

# Page 1: Stock_Price_Predictions
def stock_price_prediction():
    st.header('Stock Market Predictor')

    # User input for stock symbol
    stock_symbol = st.text_input('Enter Stock Symbol', 'GOOG')
    start = '2012-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    # Fetch data
    data = fetch_data(stock_symbol, start, end)

    # Create two columns for stock price path and stock data
    col1, col2 = st.columns(2)

    with col1:
        # Add a new graph for the stock price path over time
        st.subheader(f'{stock_symbol.strip()} Stock Price Over Time')
        fig = go.Figure()
        if 'Close' not in data.columns:
            st.error(f"Missing 'Close' column for {stock_symbol.strip()}. Please check the stock symbol.")
        else:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{stock_symbol.strip()} Close Price'))
            st.plotly_chart(fig)

    with col2:
        # Display the stock data in order of newest to oldest for each symbol
        st.subheader(f'{stock_symbol.strip()} Stock Data')
        if 'Close' in data.columns:
            stock_data = data[::-1]
            stock_data.index = stock_data.index.date  # Remove timestamp
            st.write(stock_data)

    # Create two columns for moving averages and volume analysis
    col3, col4 = st.columns(2)

    with col3:
        # Add Moving Averages with Plotly
        st.subheader('Price vs Moving Averages')
        fig = go.Figure()
        if 'Close' in data.columns:
            ma_50 = data['Close'].rolling(50).mean()
            ma_100 = data['Close'].rolling(100).mean()
            ma_200 = data['Close'].rolling(200).mean()

            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{stock_symbol.strip()} Close Price'))
            fig.add_trace(go.Scatter(x=ma_50.index, y=ma_50, mode='lines', name=f'{stock_symbol.strip()} 50-Day MA'))
            fig.add_trace(go.Scatter(x=ma_100.index, y=ma_100, mode='lines', name=f'{stock_symbol.strip()} 100-Day MA'))
            fig.add_trace(go.Scatter(x=ma_200.index, y=ma_200, mode='lines', name=f'{stock_symbol.strip()} 200-Day MA'))
            fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig)

    with col4:
        # Add Volume Analysis with Plotly
        st.subheader('Volume Analysis')
        fig = go.Figure()
        if 'Volume' not in data.columns:
            st.error(f"Missing 'Volume' column for {stock_symbol.strip()}.")
        else:
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name=f'{stock_symbol.strip()} Volume'))
            st.plotly_chart(fig)

    # Apply trading models for the symbol
    if 'Close' in data.columns:
        # Apply trading models
        ma_crossover_data = moving_average_crossover(stock_symbol.strip(), start, end)
        bb_data = bollinger_bands(stock_symbol.strip(), start, end)
        rsi_data = relative_strength_index(stock_symbol.strip(), start, end)
        sma_mr_data = sma_mean_reversion(stock_symbol.strip(), start, end)
        rf_preds, rf_data = random_forest_predictions(stock_symbol.strip(), start, end)

        # Determine final actions based on signals
        mac_action = 'Buy' if ma_crossover_data['Signal'].iloc[-1] == 1 else 'Sell' if ma_crossover_data['Signal'].iloc[-1] == -1 else 'Hold'
        bb_action = 'Buy' if bb_data['Signal'].iloc[-1] == 1 else 'Sell' if bb_data['Signal'].iloc[-1] == -1 else 'Hold'
        rsi_action = 'Buy' if rsi_data['Signal'].iloc[-1] == 1 else 'Sell' if rsi_data['Signal'].iloc[-1] == -1 else 'Hold'
        sma_action = 'Buy' if sma_mr_data['Signal'].iloc[-1] == 1 else 'Sell' if sma_mr_data['Signal'].iloc[-1] == -1 else 'Hold'
        rf_action = 'Buy' if rf_preds[-1] == 1 else 'Sell' if rf_preds[-1] == -1 else 'Hold'

        # Display model results
        st.subheader(f'Trading Signals for {stock_symbol.strip()}')
        st.write('Moving Average Crossover Signal:', mac_action)
        st.write('Bollinger Bands Signal:', bb_action)
        st.write('Relative Strength Index Signal:', rsi_action)
        st.write('SMA Mean Reversion Signal:', sma_action)
        st.write(f'Random Forest Model Signal: {rf_action}')

        # Use combine_model_outputs function to get the required data for prediction
        combined_data = combine_model_outputs(stock_symbol.strip(), start, end)
        combined_data_last_row = combined_data.iloc[[-1]]

        # Print the combined data
        st.subheader('Combined Data for Prediction')
        st.write(combined_data_last_row)

        # Prepare the last row of data for prediction
        last_row = combined_data.iloc[[-1]].drop(columns=['Future_Return', 'Target'], errors='ignore')
        prediction = model.predict(last_row)[0]
        final_decision = 'Buy' if prediction == 1 else 'Sell' if prediction == -1 else 'Hold'

        # Display final decision
        st.subheader(f'Final decision: {final_decision}')


# export FRED_API_KEY='d830b3329bbd389161db44a614feef16'
# Fetch FRED API key from environment variable
# fred_api_key = os.getenv('d830b3329bbd389161db44a614feef16')

import certifi
import ssl
import urllib
from fredapi import Fred
import pandas as pd

fred_api_key = 'd830b3329bbd389161db44a614feef16'
fred = Fred(api_key=fred_api_key)

# Configure urllib to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())

class VerifiedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, ssl_context=None, **kwargs):
        super().__init__(context=ssl_context, **kwargs)

opener = urllib.request.build_opener(VerifiedHTTPSHandler(ssl_context))
urllib.request.install_opener(opener)

# Page 2: Optimal Portfolio Allocation
def optimal_portfolio_allocation():
    def fetch_data(tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)
        return data['Adj Close']

    def fetch_macroeconomic_data(start_date, end_date):
        inflation = fred.get_series('CPIAUCSL', start_date, end_date).pct_change() * 100
        interest_rate = fred.get_series('DFF', start_date, end_date)

        inflation_df = pd.DataFrame(inflation, columns=['Inflation']).resample('M').apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan).fillna(method='ffill').fillna(method='bfill')
        interest_rate_df = pd.DataFrame(interest_rate, columns=['Interest_Rate']).resample('M').apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan).fillna(method='ffill').fillna(method='bfill')

        macroeconomic_features = pd.concat([inflation_df, interest_rate_df], axis=1)
        macroeconomic_features = macroeconomic_features.dropna(subset=['Inflation'])

        return macroeconomic_features

    def preprocess_data(data):
        if len(data) < 2:
            return pd.DataFrame()
        monthly_data = data.resample('M').last()
        if len(monthly_data) < 2:
            return pd.DataFrame()
        monthly_returns = monthly_data.pct_change().dropna()
        return monthly_returns

    def create_lagged_features(data, lags=1):
        lagged_data = pd.concat([data.shift(i).add_suffix(f'_lag_{i}') for i in range(1, lags + 1)], axis=1)
        return lagged_data

    def create_technical_indicators(data):
        indicators = pd.concat([
            data.rolling(window=3).mean().add_suffix('_mean_return'),
            data.rolling(window=3).std().add_suffix('_volatility')
        ], axis=1)
        return indicators

    def create_features(data):
        features = data.copy()
        for lag in range(1, 2):
            lagged = data.shift(lag)
            lagged.columns = [f'{col}_lag_{lag}' for col in data.columns]
            features = pd.concat([features, lagged], axis=1)

        for window in [3, 6, 12]:
            moving_avg = data.rolling(window=window).mean()
            moving_avg.columns = [f'{col}_mean_{window}m' for col in data.columns]
            features = pd.concat([features, moving_avg], axis=1)

        for window in [3, 6, 12]:
            volatility = data.rolling(window=window).std()
            volatility.columns = [f'{col}_vol_{window}m' for col in data.columns]
            features = pd.concat([features, volatility], axis=1)

        for ticker in data.columns:
            features[f'price_{ticker}'] = data[ticker]

        features = features.fillna(method='bfill')
        return features

    def concat(features, macroeconomics=None):
        features.index = features.index.normalize()
        
        if features.index.tz is not None:
            features.index = features.index.tz_convert(None)

        if macroeconomics is not None:
            macroeconomics.index = macroeconomics.index.normalize()
            if macroeconomics.index.tz is not None:
                macroeconomics.index = macroeconomics.index.tz_convert(None)
            if not macroeconomics.index.equals(features.index):
                macroeconomics = macroeconomics.resample('M').last()
            features = pd.concat([features, macroeconomics], axis=1)

        features['Inflation'].fillna(method='ffill', inplace=True)
        features['Interest_Rate'].fillna(method='ffill', inplace=True)

        return features

    def align_features_and_targets(features, returns):
        shifted_returns = returns.shift(-1).dropna()
        aligned_features = features.loc[shifted_returns.index]
        return aligned_features, shifted_returns

    def train_keras_model(X, y):
        model = Sequential()
        model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        return model

    def train_linear_model(X, y):
        model = LinearRegression().fit(X, y)
        return model

    def train_decision_tree_model(X, y):
        model = DecisionTreeRegressor().fit(X, y)
        return model

    def train_random_forest_model(X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        return model

    def align_features_and_targets_for_date(features, returns, date):
        aligned_features = features.loc[:date].dropna()
        aligned_returns = returns.loc[aligned_features.index]
        return aligned_features, aligned_returns

    def get_model_predictions_for_last_12_months(tickers, combined_features, shifted_returns):
        keras_predictions = []
        linear_predictions = []
        decision_tree_predictions = []
        random_forest_predictions = []
        actual_returns = []

        for date in combined_features.index[-13:-1]:
            keras_preds = []
            linear_preds = []
            decision_tree_preds = []
            random_forest_preds = []
            actuals = []

            for ticker in tickers:
                stock_features = combined_features.filter(regex=f'^{ticker}')
                macro_features = combined_features[['Inflation', 'Interest_Rate']]
                X_full = pd.concat([stock_features, macro_features], axis=1)
                y_full = shifted_returns[ticker]

                X, y = align_features_and_targets_for_date(X_full, y_full, date)

                keras_model = train_keras_model(X, y)
                linear_model = train_linear_model(X, y)
                decision_tree_model = train_decision_tree_model(X, y)
                random_forest_model = train_random_forest_model(X, y)

                X_last = X_full.loc[date].values.reshape(1, -1)
                keras_preds.append(keras_model.predict(X_last)[0][0])
                linear_preds.append(linear_model.predict(X_last)[0])
                decision_tree_preds.append(decision_tree_model.predict(X_last)[0])
                random_forest_preds.append(random_forest_model.predict(X_last)[0])
                actuals.append(shifted_returns[ticker].loc[date])

            keras_predictions.append(keras_preds)
            linear_predictions.append(linear_preds)
            decision_tree_predictions.append(decision_tree_preds)
            random_forest_predictions.append(random_forest_preds)
            actual_returns.append(actuals)

        return np.array(keras_predictions), np.array(linear_predictions), np.array(decision_tree_predictions), np.array(random_forest_predictions), np.array(actual_returns)

    def train_meta_model(X_meta, y_meta):
        meta_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_meta, y_meta)
        return meta_model

    def calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.04, alpha=2):
        portfolio_return = np.sum(mean_returns * weights) * 12
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
        sharpe_ratio = (alpha * portfolio_return - risk_free_rate) / portfolio_std_dev
        return -sharpe_ratio

    def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.04, alpha=2):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate, alpha)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        result = minimize(calculate_sharpe_ratio, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
        portfolio_return = np.sum(mean_returns * weights) * 12
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return portfolio_return, portfolio_std_dev, sharpe_ratio

    def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.04):
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return, portfolio_std_dev, sharpe_ratio = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std_dev
            results[2, i] = sharpe_ratio

        return results, weights_record

    def display_efficient_frontier(mean_returns, cov_matrix, optimized_weights, risk_free_rate=0.04):
        results, weights = generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=risk_free_rate)

        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
        max_sharpe_allocation = weights[max_sharpe_idx]

        min_vol_idx = np.argmin(results[1])
        sdp_min, rp_min = results[1, min_vol_idx], results[0, min_vol_idx]
        min_vol_allocation = weights[min_vol_idx]

        st.write("Maximum Sharpe Ratio Portfolio Allocation\n")
        st.write(f"Annualized Return: {rp:.2f}")
        st.write(f"Annualized Volatility: {sdp:.2f}")
        st.write(f"Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}")
        st.write(pd.DataFrame(max_sharpe_allocation, index=mean_returns.index, columns=['Allocation']).T)
        
        st.write("Minimum Volatility Portfolio Allocation\n")
        st.write(f"Annualized Return: {rp_min:.2f}")
        st.write(f"Annualized Volatility: {sdp_min:.2f}")
        st.write(f"Sharpe Ratio: {results[2, min_vol_idx]:.2f}")
        st.write(pd.DataFrame(min_vol_allocation, index=mean_returns.index, columns=['Allocation']).T)

        frontier_df = pd.DataFrame({
            'Return': results[0],
            'Volatility': results[1],
            'Sharpe Ratio': results[2]
        })

        frontier_chart = alt.Chart(frontier_df).mark_circle(size=60).encode(
            x='Volatility',
            y='Return',
            color='Sharpe Ratio',
            tooltip=['Volatility', 'Return', 'Sharpe Ratio']
        ).interactive().properties(
            title='Efficient Frontier'
        )

        st.altair_chart(frontier_chart, use_container_width=True)

    def display_results(tickers_list, returns, combined_features, shifted_returns):
        keras_predictions, linear_predictions, decision_tree_predictions, random_forest_predictions, actual_returns = get_model_predictions_for_last_12_months(tickers_list, combined_features, shifted_returns)

        meta_models = {}
        for i, ticker in enumerate(tickers_list):
            X_meta = np.hstack((keras_predictions[:, i].reshape(-1, 1), 
                                linear_predictions[:, i].reshape(-1, 1), 
                                decision_tree_predictions[:, i].reshape(-1, 1), 
                                random_forest_predictions[:, i].reshape(-1, 1)))
            y_meta = actual_returns[:, i]
            meta_models[ticker] = train_meta_model(X_meta, y_meta)
        
        last_date = pd.to_datetime('2023-06-30')
        one_month_ago = last_date - DateOffset(months=1)
        if last_date in combined_features.index:
            keras_predictions = []
            linear_predictions = []
            decision_tree_predictions = []
            random_forest_predictions = []

            for ticker in tickers_list:
                stock_features = combined_features.filter(regex=f'^{ticker}')
                macro_features = combined_features[['Inflation', 'Interest_Rate']]
                X = pd.concat([stock_features, macro_features], axis=1)
                y = shifted_returns[ticker]

                X, y = align_features_and_targets_for_date(X, y, one_month_ago)

                keras_model = train_keras_model(X, y)
                linear_model = train_linear_model(X, y)
                decision_tree_model = train_decision_tree_model(X, y)
                random_forest_model = train_random_forest_model(X, y)

                stock_features_pred = combined_features.filter(regex=f'^{ticker}')
                macro_features_pred = combined_features[['Inflation', 'Interest_Rate']]
                X_pred = pd.concat([stock_features_pred, macro_features_pred], axis=1)
                X_last = X_pred.loc[last_date].values.reshape(1, -1)
                
                keras_prediction = keras_model.predict(X_last)[0][0]
                linear_prediction = linear_model.predict(X_last)[0]
                decision_tree_prediction = decision_tree_model.predict(X_last)[0]
                random_forest_prediction = random_forest_model.predict(X_last)[0]

                keras_predictions.append(keras_prediction)
                linear_predictions.append(linear_prediction)
                decision_tree_predictions.append(decision_tree_prediction)
                random_forest_predictions.append(random_forest_prediction)

            meta_predictions = []
            for i, ticker in enumerate(tickers_list):
                X_meta_last = np.hstack((keras_predictions[i], linear_predictions[i], decision_tree_predictions[i], random_forest_predictions[i])).reshape(1, -1)
                meta_prediction = meta_models[ticker].predict(X_meta_last)[0]
                meta_predictions.append(meta_prediction)

            # Display meta model predictions
            meta_predictions_df = pd.DataFrame(meta_predictions, index=tickers_list, columns=['Meta Model Prediction'])
            st.write(f"Meta Model Predictions for {last_date.date()}:")
            st.table(meta_predictions_df)
            
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            alpha = st.session_state['alpha']
            optimized_result = optimize_portfolio(mean_returns, cov_matrix, alpha=alpha)
            optimized_weights = optimized_result.x

            portfolio_return = np.sum(mean_returns * optimized_weights)
            portfolio_std_dev = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))

            weighted_portfolio = {ticker: weight for ticker, weight in zip(tickers_list, optimized_weights)}

            # Convert dictionary to DataFrame for display
            portfolio_df = pd.DataFrame(weighted_portfolio.items(), columns=['Ticker', 'Weight'])

            st.write('Optimized Portfolio Weights:')
            st.table(portfolio_df)
            st.write('Expected Portfolio Return:', portfolio_return)
            st.write('Portfolio Standard Deviation:', portfolio_std_dev)

            # Create a DataFrame for the last 12 months of returns
            last_12_months_returns = returns.tail(12).reset_index()
            last_12_months_returns = last_12_months_returns.melt(id_vars='Date', var_name='Ticker', value_name='Return')

            # Create a line chart with Altair
            line_chart = alt.Chart(last_12_months_returns).mark_line().encode(
                x='Date:T',
                y='Return:Q',
                color='Ticker:N',
                tooltip=['Date:T', 'Ticker:N', 'Return:Q']
            ).properties(
                title='Returns of the Last 12 Months'
            )

            st.altair_chart(line_chart, use_container_width=True)

            # Display the efficient frontier
            display_efficient_frontier(mean_returns, cov_matrix, optimized_weights)
            
        else:
            st.write(f"No features available for {last_date.date()}")

    st.title('Stock Portfolio Optimizer using Modern Portfolio Theory')

    tickers = st.text_input('Enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOG)', 'AAPL, MSFT, GOOG')

    start_date = '2018-01-01'
    end_date = (datetime.today().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')

    if st.button('Fetch Data'):
        tickers_list = [ticker.strip() for ticker in tickers.split(',')]
        data = fetch_data(tickers_list, start_date, end_date)
        macro_data = fetch_macroeconomic_data(start_date, end_date)
        returns = preprocess_data(data)
        features = create_features(returns)
        combined_features = concat(features, macroeconomics=macro_data)
        aligned_features, shifted_returns = align_features_and_targets(combined_features, returns)
        
        st.session_state['tickers_list'] = tickers_list
        st.session_state['returns'] = returns
        st.session_state['combined_features'] = combined_features
        st.session_state['shifted_returns'] = shifted_returns

    if 'alpha' not in st.session_state:
        st.session_state['alpha'] = 2.0

    alpha = st.number_input('Set Alpha for Sharpe Ratio Calculation', min_value=1.0, max_value=3.0, step=0.1, value=st.session_state['alpha'], key='alpha', on_change=st.experimental_rerun)

    if 'tickers_list' in st.session_state and 'returns' in st.session_state and 'combined_features' in st.session_state and 'shifted_returns' in st.session_state:
        display_results(st.session_state['tickers_list'], st.session_state['returns'], st.session_state['combined_features'], st.session_state['shifted_returns'])

def portfolio_positions():
    st.header('Current Portfolio Positions')

    # Get positions
    positions = api.list_positions()

    # Get account details
    account = api.get_account()
    cash_balance = float(account.cash)

    # Calculate portfolio value
    portfolio_value = float(account.portfolio_value)
    equity_value = portfolio_value - cash_balance

    # Display portfolio value and cash balance
    st.write(f'**Portfolio Value:** ${portfolio_value:,.2f}')
    st.write(f'**Cash Balance:** ${cash_balance:,.2f}')
    st.write(f'**Equity Value:** ${equity_value:,.2f}')

    # Display positions
    if positions:
        position_data = []
        for position in positions:
            position_data.append({
                'Symbol': position.symbol,
                'Quantity': f"{float(position.qty):,.4f}",  # Changed to float and formatted to 4 decimal places
                'Current price': f"${float(position.current_price):,.2f}",
                'Market value': f"${float(position.market_value):,.2f}",
                'Cost basis': f"${float(position.cost_basis):,.2f}",
                'Unrealized profit/loss': f"${float(position.unrealized_pl):,.2f}"
            })
        st.table(position_data)
    else:
        st.write('No positions currently held.')

    # Prepare data for pie chart
    symbols = [position['Symbol'] for position in position_data]
    values = [float(position['Market value'].replace('$', '').replace(',', '')) for position in position_data]

    # Pie chart
    st.subheader('Investment Distribution')
    fig = go.Figure(data=[go.Pie(labels=symbols, values=values, textinfo='label+percent')])
    st.plotly_chart(fig)

    # Button to refresh the portfolio position
    if st.button('Refresh Portfolio'):
        st.experimental_rerun()


# Main App
page = st.sidebar.selectbox("Choose a page", ["Short term strategies", "Optimal Portfolio Allocation", "Current Portfolio Positions"])

if page == "Short term strategies":
    stock_price_prediction()
elif page == "Optimal Portfolio Allocation":
    optimal_portfolio_allocation()
elif page == "Current Portfolio Positions":
    portfolio_positions()
