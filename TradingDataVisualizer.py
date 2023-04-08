import ta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_macd(df):
    # Convert date string to datetime object and set as index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Calculate the MACD
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["Close"]).macd_signal()
    df["macd_diff"] = ta.trend.MACD(df["Close"]).macd_diff()
 
    # Calculate the Moving Averages
    df["ma7"] = df["Close"].rolling(window=7).mean()
    df["ma25"] = df["Close"].rolling(window=25).mean()
    df["ma99"] = df["Close"].rolling(window=99).mean()
    
    # Create the figure and axis objects
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot the close price in the first row
    axs[0].plot(df["Close"], label="Close Price")
    axs[0].plot(df["ma7"], label="MA 7", color="purple", linestyle=":")
    axs[0].plot(df["ma25"], label="MA 25", color="orange", linestyle=":")
    axs[0].plot(df["ma99"], label="MA 99", color="gray", linestyle=":")
    axs[0].legend(loc="best")
    axs[0].set_title("Close Price")
    axs[0].grid(True)
    
    # Plot the MACD in the second row
    cmap = cm.get_cmap("RdYlGn")
    normalize = lambda x: (x - df["macd_diff"].min()) / (df["macd_diff"].max() - df["macd_diff"].min())
    colors = [cmap(normalize(x)) for x in df["macd_diff"]]
    axs[1].bar(df.index, df["macd_diff"], label="MACD Difference", color=colors)
    axs[1].plot(df["macd"], label="MACD", color="red", linestyle="--")
    axs[1].plot(df["macd_signal"], label="MACD Signal", color="blue")
    axs[1].legend(loc="best")
    axs[1].set_title("MACD")
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# load data into a pandas dataframe
df = pd.read_csv("data2\Binance_ETHUSDT_d.csv")

# plot the MACD
plot_macd(df)
s = 9

# # calculate MACD using the ta library
# df["macd"] = ta.trend.MACD(df["Close"]).macd
# df["macd_signal"] = ta.trend.MACD(df["Close"]).macd_signal
# df["macd_diff"] = ta.trend.MACD(df["Close"]).macd_diff

# # display the results
# print(df.tail())