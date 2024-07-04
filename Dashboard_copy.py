import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime
import alpaca_trade_api as tradeapi
import joblib
from trading_models import moving_average_crossover, bollinger_bands, relative_strength_index, sma_mean_reversion, combine_model_outputs, random_forest_predictions

# Alpaca API credentials (replace with your own)
ALPACA_API_KEY = 'PKVGHKG5MBR1NFE0MWZ9'
ALPACA_SECRET_KEY = 'LKYed9zuEdMvvqpck2y9mcYcd7YFPKlza8YCH4e9'
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


# Function to download stock data
def download_data(stock_list, start, end):
    data = yf.download(stock_list, start=start, end=end)['Adj Close']
    return data

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(data):
    # Calculate weekly returns
    returns = data.pct_change().dropna()

    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Define portfolio parameters
    num_assets = len(data.columns)
    num_portfolios = 10000  # Number of portfolios to simulate
    target_returns = np.linspace(0, 0.5, num_portfolios)  # Range of target returns
    weights = np.random.rand(num_portfolios, num_assets)
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights to sum up to 1 for each portfolio

    # Calculate portfolio expected returns and volatilities
    portfolio_returns = np.dot(weights, returns.mean()) * 52  # Annualized expected returns
    portfolio_volatilities = np.sqrt(np.diag(np.dot(weights, np.dot(cov_matrix, weights.T)))) * np.sqrt(52)  # Annualized volatilities

    # Calculate Sharpe Ratio for each portfolio
    risk_free_rate = 0  # Assume risk-free rate as 0 for simplicity
    sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_volatilities

    # Identify portfolio with maximum Sharpe Ratio (optimal portfolio)
    optimal_portfolio_idx = np.argmax(sharpe_ratios)
    optimal_portfolio_return = portfolio_returns[optimal_portfolio_idx]
    optimal_portfolio_volatility = portfolio_volatilities[optimal_portfolio_idx]
    optimal_portfolio_weights = weights[optimal_portfolio_idx]
    optimal_portfolio_sharpe_ratio = sharpe_ratios[optimal_portfolio_idx]

    return portfolio_returns, portfolio_volatilities, optimal_portfolio_return, optimal_portfolio_volatility, optimal_portfolio_weights, optimal_portfolio_sharpe_ratio

# Function to plot efficient frontier
def plot_efficient_frontier(portfolio_volatilities, portfolio_returns, optimal_portfolio_volatility, optimal_portfolio_return):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o', cmap='viridis')
    ax.scatter(optimal_portfolio_volatility, optimal_portfolio_return, marker='x', color='red', s=100, label='Optimal Portfolio')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    fig.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Streamlit app function
def optimal_portfolio_allocation():
    st.header('Long-Term Investment Strategy: Optimal Portfolio Allocation')

    # User input for multiple stock symbols
    stocks = st.text_area('Enter Stock Symbols (comma separated)', 'AAPL, MSFT, GOOGL, AMZN, META')
    stock_list = [stock.strip() for stock in stocks.split(',')]
    start = '2012-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    # Download stock data
    data = download_data(stock_list, start, end)

    # Calculate portfolio metrics
    portfolio_returns, portfolio_volatilities, optimal_portfolio_return, optimal_portfolio_volatility, optimal_portfolio_weights, optimal_portfolio_sharpe_ratio = calculate_portfolio_metrics(data)

    # Display optimal portfolio return and volatility
    st.subheader('Optimal Portfolio Metrics')
    st.write(f'- Optimal Portfolio Return: {optimal_portfolio_return:.2%} (annualized)')
    st.write(f'- Optimal Portfolio Volatility: {optimal_portfolio_volatility:.2%} (annualized)')
    st.write(f'- Optimal Portfolio Sharpe Ratio: {optimal_portfolio_sharpe_ratio:.2f}')

    # Plot efficient frontier
    st.subheader('Efficient Frontier')
    plot_efficient_frontier(portfolio_volatilities, portfolio_returns, optimal_portfolio_volatility, optimal_portfolio_return)

    # Display optimal portfolio weights
    st.subheader('Optimal Portfolio Weights')
    optimal_weights_df = pd.DataFrame({
        'Stock': stock_list,
        'Allocation': optimal_portfolio_weights * 100
    })
    optimal_weights_df.set_index('Stock', inplace=True)
    optimal_weights_df.sort_values(by='Allocation', ascending=False, inplace=True)
    optimal_weights_df['Allocation'] = optimal_weights_df['Allocation'].map('{:.2f}%'.format)
    st.table(optimal_weights_df)

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
