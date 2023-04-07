import pandas as pd
from multiprocessing import Pool
import datetime
import glob, csv, os


class TradingBot:
    def __init__(self, signal_function, initial_balance=100000):
        self.signal_function = signal_function
        self.balance = initial_balance
        self.holdings = 0
    
    def trade(self, signal):
        if signal == "Buy" and self.balance > 0:
            # Buy half of the balance
            self.holdings += self.balance / 2
            # Sell the other half
            self.balance -= self.holdings
        elif signal == "Sell" and self.holdings > 0:
            self.balance += self.holdings
            self.holdings = 0
        else:
            pass
        print("Signal:",signal ,"Balance:", self.balance, "Holdings:", self.holdings)

def generate_signal_moving_average(df, short_window=7, long_window=25):
    """
    Generate a signal based on Moving Averages
    :param df: DataFrame containing the asset's close price
    :param short_window: The number of days used for the short-term moving average
    :param long_window: The number of days used for the long-term moving average
    :return: List of signals, where 1 indicates a buy signal, -1 indicates a sell signal, and 0 indicates hold
    """
    # Calculate the moving averages
    df['short_ma'] = df['Close'].rolling(window=short_window).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window).mean()
    
    # Generate the signals
    signals = []
    for i in range(len(df)):
        if (df['short_ma'].iloc[i] > df['long_ma'].iloc[i]) and (df['short_ma'].iloc[i - 1] < df['long_ma'].iloc[i - 1]):
            signals.append(1) # Buy signal
        elif (df['short_ma'].iloc[i] < df['long_ma'].iloc[i]) and (df['short_ma'].iloc[i - 1] > df['long_ma'].iloc[i - 1]):
            signals.append(-1) # Sell signal
        else:
            signals.append(0) # Hold signal
            
    return signals

def generate_signal_bollinger_bands(df, window=20, num_std=2):
    """
    Generate a signal based on Bollinger Bands
    :param df: DataFrame containing the asset's close price
    :param window: The number of days used for calculating the rolling mean and standard deviation
    :param num_std: The number of standard deviations used for calculating the upper and lower Bollinger Bands
    :return: List of signals, where 1 indicates a buy signal, -1 indicates a sell signal, and 0 indicates hold
    """
    # Calculate the rolling mean and standard deviation
    df['rolling_mean'] = df['Close'].rolling(window=window).mean()
    df['rolling_std'] = df['Close'].rolling(window=window).std()
    
    # Calculate the upper and lower Bollinger Bands
    df['upper_band'] = df['rolling_mean'] + num_std * df['rolling_std']
    df['lower_band'] = df['rolling_mean'] - num_std * df['rolling_std']
    
    # Generate the signals
    signals = []
    for i in range(len(df)):
        if df['Close'].iloc[i] > df['upper_band'].iloc[i]:
            signals.append(-1) # Sell signal
        elif df['Close'].iloc[i] < df['lower_band'].iloc[i]:
            signals.append(1) # Buy signal
        else:
            signals.append(0) # Hold signal
            
    return signals

def generate_signal_rsi(df, window=14):
    """
    Generate a signal based on the Relative Strength Index (RSI)
    :param df: DataFrame containing the asset's close price
    :param window: The number of days used for calculating the RSI
    :return: List of signals, where 1 indicates a buy signal, -1 indicates a sell signal, and 0 indicates hold
    """
    # Calculate the differences between consecutive days
    delta = df['Close'].diff()
    
    # Calculate the gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate the average gains and losses over the specified window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate the relative strength
    rs = avg_gain / avg_loss
    
    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Generate the signals
    signals = []
    for i in range(len(df)):
        if rsi.iloc[i] >= 50:
            signals.append(1) # Buy signal
        else:
            signals.append(-1) # Sell signal
            
    return signals

def generate_smart_signal(df, window=14, n_std=2):
    """
    Generates a smart signal based on Bollinger Bands, Moving Averages, and the Relative Strength Index
    :param df: DataFrame containing the asset's price data
    :param window: Window size for the Moving Averages and RSI
    :param n_std: Number of standard deviations for the Bollinger Bands
    :return: Series containing the smart signals
    """
    # Calculate the Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA25'] = df['Close'].rolling(window=25).mean()
    df['MA99'] = df['Close'].rolling(window=99).mean()
    
    # Calculate the Bollinger Bands
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['upper_band'] = rolling_mean + n_std * rolling_std
    df['lower_band'] = rolling_mean - n_std * rolling_std
    
    # Calculate the Relative Strength Index
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().abs()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate the smart signals
    signals = []
    for i in range(len(df)):
        if df.at[i, 'Close'] < df.at[i, 'lower_band'] and df.at[i, 'RSI'] < 30:
            signals.append(1)
        elif df.at[i, 'Close'] > df.at[i, 'upper_band'] and df.at[i, 'RSI'] > 70:
            signals.append(-1)
        else:
            signals.append(0)
            
    return pd.Series(signals, index=df.index)

def run_trading_bot(data):
    bot = TradingBot(signal_function=generate_signal_rsi)
    for i in range(1, len(data)):
        signal = bot.signal_function(data.iloc[:i])
        bot.trade(signal)
    print("Final balance:", bot.balance)

def trade_bot(df, generate_signal, initial_capital=100000):
    """
    A trading bot that makes decisions based on the signals generated by a specified signal function
    :param df: DataFrame containing the asset's price data
    :param generate_signal: Function that generates the signals
    :param initial_capital: The initial capital for the trading bot
    :return: DataFrame containing the trades made by the trading bot
    """
    #print initial capital
    #print("Initial capital:", initial_capital)

    #print("The signal function name is:", generate_signal.__name__)

    fee = 0.0005  #fee in percentage

    # Generate the signals
    signals = generate_signal(df)
    
    # Initialize the trades DataFrame
    trades = pd.DataFrame({'timestamp': df.index, 'signal': signals})
    
    # Initialize the current position and capital
    position = 0
    capital = initial_capital
    
    # Iterate through the trades
    for i, trade in trades.iterrows():
        # Buy if the signal is 1
        if trade['signal'] == 1:
            if position == 0:
                #subtract the fee
                capital = capital - (capital * fee)
                position = capital / df.at[i, 'Close']
                capital = 0
                
        # Sell if the signal is -1
        elif trade['signal'] == -1:
            if position != 0:
                capital = position * df.at[i, 'Close']
                #subtract the fee
                capital = capital - (capital * fee)
                position = 0
                
        # Update the trades DataFrame
        trades.at[i, 'position'] = position
        trades.at[i, 'capital'] = capital
        
    # Calculate the portfolio value
    trades['value'] = trades['capital'] + trades['position'] * df['Close']    

    # Calculate the returns
    trades['returns'] = trades['value'].pct_change()
    #print("Final balance:", trades['value'].iloc[-1], "Return: ", trades['returns'].iloc[-1] )

    return trades

def evaluate_strategy(df, trade_bot, generate_signal, short_window_options, long_window_options, initial_capital=100000):
    best_result = float("-inf")
    best_strategy = None

    #sort the data frame by date
    df = df.sort_values(by='Date')

    for short_window in short_window_options:
        for long_window in long_window_options:
            def current_strategy(df):
                return generate_signal(df, short_window=short_window, long_window=long_window)
            result = trade_bot(df, current_strategy, initial_capital=initial_capital)
            final_balance = result['value'].iloc[-1]
            
            if final_balance > best_result:
                best_result = final_balance
                #format best_result to 2 decimal places
                best_result = "{:.2f}".format(best_result)
                best_strategy = f"Short window: {short_window}, Long window: {long_window}"
    return best_result, best_strategy

# def run_strategy_evaluation(df, trade_bot, generate_signal, initial_capital=100000):
#     short_window_options = range(5, 101, 1)
#     long_window_options = range(25, 505, 5)

#     short_window_options = [7, 14, 21]
#     long_window_options = [25, 50, 100]

#     pool = Pool(processes=10)
#     results = [pool.apply_async(evaluate_strategy, (df, trade_bot, generate_signal, [short_window], [long_window], initial_capital))
#                for short_window in short_window_options
#                for long_window in long_window_options]

#     best_result = None
#     best_strategy = None
#     for result in results:
#         res, strategy = result.get()
#         if best_result is None or res > best_result:
#             best_result = res
#             best_strategy = strategy

#     print("Best result:", best_result)
#     print("Best strategy:", best_strategy)

def run_strategy_evaluation(name, df, trade_bot, generate_signal, initial_capital=100000):
    # short_window_options = [7, 14, 21]
    # long_window_options = [25, 50, 100]

    short_window_options = range(5, 101, 1)
    long_window_options = range(25, 505, 5)
    
    results = []
    strategies = []
    
    pool = Pool(processes=14)
    for short_window in short_window_options:
        for long_window in long_window_options:
            result = pool.apply_async(evaluate_strategy, (df, trade_bot, generate_signal, [short_window], [long_window], initial_capital))
            results.append(result)
            strategies.append((short_window, long_window))
            
    pool.close()
    pool.join()
    
    best_result = None
    best_strategy = None
    for i, result in enumerate(results):
        res, strategy = result.get()
        if best_result is None or res > best_result:
            best_result = res
            best_strategy = strategies[i]

    print("Best result:", best_result)
    print("Best strategy:", best_strategy)

    #Calculate profit when using the intitial capital to buy stocks in the beginning and holdig them until the end
    nb_stocks = initial_capital / df['Close'].iloc[-1]
    just_holding_profit = nb_stocks * df['Close'].iloc[0]
    #format the profit to 2 decimals
    just_holding_profit = "{:.2f}".format(just_holding_profit)
    print("just_holding_profit:", just_holding_profit)
    
    with open(f"{name}_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Short Window", "Long Window", "Result", "Just Holding Profit"])
        for i, result in enumerate(results):
            res = result.get()[0]
            writer.writerow([strategies[i][0], strategies[i][1], res, just_holding_profit])


def find_best_result(files_dir):
    best_result = 0
    best_file = None
    
    for filename in os.listdir(files_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(files_dir, filename)
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) # skip the header row
                for row in reader:
                    result = float(row[2])
                    if result > best_result:
                        best_result = result
                        best_file = filename
    
    return best_file, best_result

import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_csv(file_path):
    x, y, z = [], [], []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip the header row
        results = []
        for row in reader:
            short_window = int(row[0])
            long_window = int(row[1])
            result = float(row[2])
            
            x.append(short_window)
            y.append(long_window)
            z.append(result)
            results.append((short_window, long_window, result))
    
    results.sort(key=lambda x: x[2], reverse=True)
    best_results = results[:10]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('Short Window')
    ax.set_ylabel('Long Window')
    ax.set_zlabel('Result')
    
    for result in best_results:
        ax.text(*result[:3], "%.2f" % result[2], color='red')
    
    plt.show()

import pandas as pd
import numpy as np

def find_best_windows(dfs):
    results = []
    for df in dfs:
        mean_result = df.groupby(['Short Window', 'Long Window'])['Result'].mean().reset_index()
        best_result = mean_result.loc[mean_result['Result'].idxmax()]
        results.append({'Short Window': best_result['Short Window'],
                        'Long Window': best_result['Long Window'],
                        'Best Mean Result': best_result['Result']})
    return pd.DataFrame(results, columns=['Short Window', 'Long Window', 'Best Mean Result'])


def visualize_signal_moving_average(df, signals, short_window=7, long_window=25):
    """
    Visualize the signals generated by generate_signal_moving_average()
    :param df: DataFrame containing the asset's close price
    :param signals: List of signals generated by generate_signal_moving_average()
    :param short_window: The number of days used for the short-term moving average
    :param long_window: The number of days used for the long-term moving average
    """
    # Plot the stock value
    plt.plot(df['Close'], label='Close Price')
    
    # Plot the moving averages
    plt.plot(df['short_ma'], label=f'short_ma ({short_window})')
    plt.plot(df['long_ma'], label=f'long_ma ({long_window})')
    
    # Plot the buy signals
    buy_indexes = [i for i, signal in enumerate(signals) if signal == 1]
    plt.scatter(buy_indexes, df.loc[buy_indexes, 'Close'], marker='^', color='green', label='Buy')
    
    # Plot the sell signals
    sell_indexes = [i for i, signal in enumerate(signals) if signal == -1]
    plt.scatter(sell_indexes, df.loc[sell_indexes, 'Close'], marker='v', color='red', label='Sell')
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Stock Value and Trading Signals')
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    # load data into a pandas dataframe
    #df = pd.read_csv("data2\Binance_BTCUSDT_d.csv")
    
    #list of signal functions
    #signal_functions = [generate_signal_bollinger_bands, generate_signal_moving_average, generate_signal_rsi, generate_smart_signal]

    #run trading bot for each signal function
    # for generate_signal in signal_functions:
    #     trade_bot(df, generate_signal)

    initial_capital=100000

    #run_strategy_evaluation(df, trade_bot, generate_signal_moving_average, initial_capital=initial_capital)

    # dataframes = []
    # csv_files = glob.glob("*_results.csv")
    # for csv_file in csv_files:
    #     now = datetime.datetime.now()
    #     #get the name of the file without the extension
    #     name = os.path.splitext(os.path.basename(csv_file))[0]
    #     print("---------------------------------")
    #     print("Processing", name)
    #     df = pd.read_csv(csv_file)

        # #sort dataframe by "Result" column
        # df.sort_values(by=['Result'], inplace=True, ascending=False)
        # top_10 = df.head(10)
        # print(top_10)

        # #extract range of short and long windows
        # short_window_options = range(top_10['Short Window'].min(), top_10['Short Window'].max()+1, 1)
        # long_window_options = range(top_10['Long Window'].min(), top_10['Long Window'].max()+1, 1)
        # print("Short window options:", short_window_options)
        # print("Long window options:", long_window_options)

        # run_strategy_evaluation(name, df, trade_bot, generate_signal_moving_average, initial_capital=initial_capital)
        # print("Time elapsed:", datetime.datetime.now() - now)

        #dataframes.append(df)
    
    #find the best windows for each strategy
    # best_windows = find_best_windows(dataframes)
    # print(best_windows)

    # best_file, best_result = find_best_result(".\\")+
    # print("Best file:", best_file)
    # print("Best result:", best_result)

    df = pd.read_csv("./data2/Binance_IDEXUSDT_d.csv")
    signals = generate_signal_moving_average(df, short_window=66, long_window=65)
    visualize_signal_moving_average(df, signals, short_window=66, long_window=65)



