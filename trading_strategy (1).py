
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# Load or Generate Stock Data
def generate_stock_data(num_days=1000, initial_price=100, volatility=2):
    np.random.seed(42)
    price_changes = 1 + np.random.randn(num_days) * (volatility / 100)
    stock_prices = initial_price * np.cumprod(price_changes)
    date_rng = pd.date_range(start='2020-01-01', periods=num_days, freq='D')
    stock_data = pd.DataFrame(date_rng, columns=['date'])
    stock_data['price'] = stock_prices
    stock_data.set_index('date', inplace=True)
    return stock_data

stock_data = generate_stock_data()

# Compute Technical Indicators
def compute_technical_indicators(data):
    # MACD and Signal Line
    short_window = 12
    long_window = 26
    signal_window = 9
    data['short_mavg'] = data['price'].ewm(span=short_window, adjust=False).mean()
    data['long_mavg'] = data['price'].ewm(span=long_window, adjust=False).mean()
    data['macd'] = data['short_mavg'] - data['long_mavg']
    data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
    
    # RSI
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    rolling_mean = data['price'].rolling(window=20).mean()
    rolling_std = data['price'].rolling(window=20).std()
    data['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    data['bollinger_lower'] = rolling_mean - (rolling_std * 2)
    return data

stock_data = compute_technical_indicators(stock_data)

# Implement Strategy Logic and Backtest
def backtest_strategy(data, initial_capital=10000.0):
    data['buy_signal'] = (data['macd'] > data['signal_line']) &                          (data['rsi'] > 30) &                          (data['price'] <= data['bollinger_lower'])
    data['sell_signal'] = (data['macd'] < data['signal_line']) |                           (data['rsi'] < 70) |                           (data['price'] >= data['bollinger_upper'])
    data['positions'] = np.where(data['buy_signal'], 1, np.where(data['sell_signal'], -1, 0))
    data['positions'] = data['positions'].diff().fillna(0)
    
    cash = initial_capital
    stock_quantity = 0
    portfolio_value = []
    for date, row in data.iterrows():
        if row['positions'] == 1 and cash > 0:
            stock_quantity = cash / row['price']
            cash = 0
        elif row['positions'] == -1 and stock_quantity > 0:
            cash = stock_quantity * row['price']
            stock_quantity = 0
        current_value = cash + stock_quantity * row['price']
        portfolio_value.append(current_value)
    data['portfolio_value'] = portfolio_value
    return data

stock_data = backtest_strategy(stock_data)

# Plotting
def plot_results(data):
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price', color='tab:blue')
    ax1.plot(data.index, data['price'], color='tab:blue', label='Stock Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.plot(data[data['buy_signal']].index, data['price'][data['buy_signal']], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    ax1.plot(data[data['sell_signal']].index, data['price'][data['sell_signal']], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Portfolio Value', color='tab:orange')  
    ax2.plot(data.index, data['portfolio_value'], color='tab:orange', label='Portfolio Value')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    plt.title('Stock Price and Portfolio Value Over Time')
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_results(stock_data)
