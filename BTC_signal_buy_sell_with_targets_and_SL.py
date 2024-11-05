import pandas as pd
import numpy as np
import ccxt  # For fetching cryptocurrency market data
import datetime
import matplotlib.pyplot as plt
import time


# Function to fetch historical market data
def fetch_historical_data(symbol, timeframe):
    exchange = ccxt.binance()
    since = exchange.parse8601('2017-01-01T00:00:00Z')
    all_ohlcv = []
    while since < exchange.milliseconds():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if not ohlcv:
            break
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)
        # Sleep to respect rate limits
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


# Function to calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['close'].ewm(span=short_window, min_periods=1).mean()
    long_ema = data['close'].ewm(span=long_window, min_periods=1).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# Function to generate buy/sell signals based on multiple indicators
def generate_signals(data, ma_windows):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['close']
    signals['volume'] = data['volume']

    for window in ma_windows:
        signals[f'ma_{window}'] = data['close'].rolling(window=window, min_periods=1).mean()

    # Calculate MACD
    signals['macd_line'], signals['macd_signal'], signals['macd_histogram'] = calculate_macd(data)

    # Calculate 20-day moving average of volume
    signals['volume_mavg'] = data['volume'].rolling(window=20, min_periods=1).mean()

    # Generate buy/sell signals
    signals['signal'] = 0.0
    signals.loc[
        (signals[f'ma_{ma_windows[0]}'] > signals[f'ma_{ma_windows[1]}']) &
        (signals[f'ma_{ma_windows[1]}'] > signals[f'ma_{ma_windows[2]}']) &
        (signals['macd_line'] > signals['macd_signal']) &
        (signals['volume'] > signals['volume_mavg']),
        'signal'
    ] = 1.0
    signals.loc[
        (signals[f'ma_{ma_windows[0]}'] < signals[f'ma_{ma_windows[1]}']) &
        (signals[f'ma_{ma_windows[1]}'] < signals[f'ma_{ma_windows[2]}']) &
        (signals['macd_line'] < signals['macd_signal']) &
        (signals['volume'] > signals['volume_mavg']),
        'signal'
    ] = -1.0

    signals['positions'] = signals['signal'].diff().fillna(0)  # Fill NaN values with 0

    return signals

# Function to plot signals and trade results
def plot_signals_and_trades(data, signals, trade_results, portfolio_values):
    fig, axs = plt.subplots(4, figsize=(15, 15), sharex=True)

    # Plot price and moving averages
    axs[0].plot(data.index, data['close'], label='Price', color='blue')
    for window in [5, 8, 13]:
        axs[0].plot(data.index, signals[f'ma_{window}'], label=f'MA {window}', linestyle='--')
    axs[0].legend()

    # Plot MACD
    axs[1].plot(data.index, signals['macd_line'], label='MACD Line', color='blue')
    axs[1].plot(data.index, signals['macd_signal'], label='MACD Signal Line', color='orange')
    axs[1].bar(data.index, signals['macd_histogram'], label='MACD Histogram', color='gray')
    axs[1].legend()

    # Plot volume with green and red bars
    close_above_open = data['close'] > data['open']
    volume_color = np.where(close_above_open, 'g', 'r')
    axs[2].bar(data.index, data['volume'], color=volume_color, label='Volume')
    axs[2].legend()

    # Plot portfolio value separately
    axs[3].plot(data.index, portfolio_values, label='Portfolio Value', color='blue')
    axs[3].legend()

    # Set titles and labels
    axs[0].set_title('Price and Moving Averages', fontsize=14)
    axs[0].set_ylabel('Price', fontsize=12)
    axs[1].set_title('MACD', fontsize=14)
    axs[1].set_ylabel('MACD', fontsize=12)
    axs[2].set_title('Volume', fontsize=14)
    axs[2].set_ylabel('Volume', fontsize=12)
    axs[3].set_title('Portfolio Value Over Time', fontsize=14)
    axs[3].set_xlabel('Date', fontsize=12)
    axs[3].set_ylabel('Portfolio Value', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot trade results as a pie chart
    win_rate = trade_results.count('win')
    loss_rate = trade_results.count('loss')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie([win_rate, loss_rate], labels=['Win', 'Loss'], colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Trade Win/Loss Rate')
    plt.show()


# Function to calculate stop-loss based on risk-reward ratio
def calculate_stop_loss(signals, rr_ratio):
    if signals['positions'].iloc[-1] == 1.0:  # Long trade
        entry_price = signals['price'].iloc[-1]
        stop_loss = entry_price - (entry_price * rr_ratio / 100)
    elif signals['positions'].iloc[-1] == -1.0:  # Short trade
        entry_price = signals['price'].iloc[-1]
        stop_loss = entry_price + (entry_price * rr_ratio / 100)
    else:
        stop_loss = None  # No trade position

    return stop_loss



# Function to simulate trades based on signals and risk-reward ratio
def simulate_trades(data, signals, initial_balance=1000, rr_ratio=2):
    balance = initial_balance
    trade_results = []
    portfolio_values = []
    open_trade = None
    trades = []

    for i in range(len(signals)):
        if signals['positions'].iloc[i] == 1.0:
            open_trade = {'entry': signals['price'].iloc[i], 'type': 'buy', 'date': signals.index[i]}
        elif signals['positions'].iloc[i] == -1.0:
            open_trade = {'entry': signals['price'].iloc[i], 'type': 'sell', 'date': signals.index[i]}

        if open_trade:
            # Calculating target and stop loss
            if open_trade['type'] == 'buy':
                target = open_trade['entry'] * (1 + rr_ratio / 100)
                stop_loss = open_trade['entry'] * (1 - rr_ratio / (2 * 100))
            else:  # open_trade['type'] == 'sell'
                target = open_trade['entry'] * (1 - rr_ratio / 100)
                stop_loss = open_trade['entry'] * (1 + rr_ratio / (2 * 100))

            # Implementing trade logic
            for j in range(i + 1, len(signals)):
                if (open_trade['type'] == 'buy' and (
                        signals['price'].iloc[j] >= target or signals['price'].iloc[j] <= stop_loss)) or \
                        (open_trade['type'] == 'sell' and (
                                signals['price'].iloc[j] <= target or signals['price'].iloc[j] >= stop_loss)):
                    result = 'win' if (open_trade['type'] == 'buy' and signals['price'].iloc[j] >= target) or (
                            open_trade['type'] == 'sell' and signals['price'].iloc[j] <= target) else 'loss'

                    # Update exit price based on trade type
                    if open_trade['type'] == 'buy':
                        exit_price = max(open_trade['entry'], signals['price'].iloc[j])
                    else:
                        exit_price = min(open_trade['entry'], signals['price'].iloc[j])

                    trade_result = balance * (rr_ratio / 100 if result == 'win' else -rr_ratio / 200)
                    balance += trade_result
                    trade_results.append(result)
                    portfolio_values.append(balance)

                    # Add trade data to trades list
                    trades.append({
                        'Date': open_trade['date'],
                        'Entry': open_trade['entry'],
                        'Type': open_trade['type'].capitalize(),
                        'Exit': exit_price,
                        'Target': target,
                        'Stop Loss': stop_loss,
                        'Result': result
                    })

                    open_trade = None
                    break
            else:
                portfolio_values.append(balance)
        else:
            portfolio_values.append(balance)

    # Check for any open trade at the end of the simulation
    if open_trade:
        trades.append({
            'Date': open_trade['date'],
            'Entry': open_trade['entry'],
            'Type': open_trade['type'].capitalize(),
            'Exit': None,
            'Target': target,
            'Stop Loss': stop_loss,
            'Result': 'waiting'
        })

    return trade_results, portfolio_values, trades



def main():
    # Parameters
    symbol = 'BTC/USDT'  # Trading pair
    timeframe = '1d'  # Daily timeframe
    ma_windows = [5, 8, 13]  # Moving average windows
    initial_balance = 1000  # Initial balance for trade simulation
    rr_ratios = [1, 2, 3]  # Range of RR ratios to test

    # Fetch historical data from 2020 to present
    data = fetch_historical_data(symbol, timeframe)

    # Generate buy/sell signals
    signals = generate_signals(data, ma_windows)

    best_rr_ratio = None
    best_stop_loss = None
    best_win_rate = 0
    results = []
    all_trades = []

    try:
        # Open Excel writer
        with pd.ExcelWriter('targets.xlsx') as writer:

            # Iterate over RR ratios
            for rr_ratio in rr_ratios:
                # Calculate stop-loss
                stop_loss = calculate_stop_loss(signals, rr_ratio)

                # Simulate trades
                trade_results, portfolio_values, trades = simulate_trades(data, signals, initial_balance, rr_ratio)

                # Calculate win rate only if there are trades executed
                if trade_results:
                    win_rate = trade_results.count('win') / len(trade_results) * 100  # Convert to percentage
                    results.append({'RR Ratio': rr_ratio, 'Win Rate (%)': win_rate, 'Stop Loss': stop_loss})
                    # Update best RR ratio and stop-loss if win rate improves
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_rr_ratio = rr_ratio
                        best_stop_loss = stop_loss

                    # Append trades to the all_trades list
                    all_trades.extend(trades)

            if best_rr_ratio is not None:
                print("Best RR Ratio:", best_rr_ratio)
                print("Best Stop Loss:", best_stop_loss)
                print(f"Best Win Rate: {best_win_rate:.2f}%")
            else:
                print("No trades executed for any RR ratio.")

            # Create DataFrames from the results and trades data and save to Excel
            if results:
                results_df = pd.DataFrame(results)
                results_df.to_excel(writer, sheet_name='Trade Results', index=False)
            else:
                pd.DataFrame({'Message': ['No results to display']}).to_excel(writer, sheet_name='Trade Results',
                                                                              index=False)

            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
            else:
                pd.DataFrame({'Message': ['No trades executed']}).to_excel(writer, sheet_name='Trades', index=False)

        # Plot the signals and trades
        plot_signals_and_trades(data, signals, trade_results, portfolio_values)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
