import pandas as pd
import ta

class TradingInterface:
    def __init__(self, exchange):
        self.exchange = exchange

    def get_ticker(self, symbol):
        # get ticker data for given symbol from exchange
        ticker = self.exchange.get_ticker(symbol)
        return ticker

    def get_order_book(self, symbol):
        # get order book data for given symbol from exchange
        pass

    def place_order(self, order):
        # place an order on the exchange
        self.exchange.execute_order(order)        

    def cancel_order(self, order_id):
        # cancel an order on the exchange
        pass

class TradingEngine:
    def __init__(self, trading_interface, trading_algorithm):
        self.trading_interface = trading_interface
        self.trading_algorithm = trading_algorithm

        self.__init_data()

    def __init_data(self):
        # initialize data for trading algorithm
        #get first 100 tickers
        for i in range(100):
            ticker = self.trading_interface.get_ticker("ETHUSDT")
            self.trading_algorithm.add_ticker(ticker)

    def run(self):
        is_running = True
        # main trading loop
        while is_running:
            # get ticker data
            ticker = self.trading_interface.get_ticker("ETHUSDT")

            # get order book data
            order_book = self.trading_interface.get_order_book("ETHUSDT")

            # run trading algorithm
            result = self.trading_algorithm.calculate(ticker, order_book)

            # check if the algorithm wants to place an order
            if result == "BUY":
                # place a buy order
                self.place_buy_order("ETHUSDT", 1)
            elif result == "SELL":
                # place a sell order
                self.place_sell_order("ETHUSDT", 1)           

            # check if to stop the trading loop

            

    def place_buy_order(self, symbol, amount):
        # place a buy order for given symbol and amount
        order = {"symbol": symbol, "type": "BUY", "amount": amount}
        self.trading_interface.place_order(order)
        pass

    def place_sell_order(self, symbol, amount):
        # place a sell order for given symbol and amount
        order = {"symbol": symbol, "type": "SELL", "amount": amount}
        self.trading_interface.place_order(order)
        pass

    def check_order_status(self, order_id):
        # check the status of an order
        pass

class Exchange:
    def __init__(self, name):
        self.name = name

    def get_api_key(self):
        # get the API key for the exchange
        pass

    def get_secret_key(self):
        # get the secret key for the exchange
        pass

    def get_exchange_info(self):
        # get information about the exchange, such as supported symbols and trading fees
        pass

    def get_balance(self, symbol):
        # get the balance of a specific symbol in the user's account
        pass

    def execute_order(self, order):
        # execute an order on the exchange
        pass

    def cancel_order(self, order_id):
        # cancel an order on the exchange
        pass

class CsvExchange(Exchange):
    def __init__(self, name, csv_file):
        super().__init__(name)
        self.csv_file = csv_file # csv file containing historical data
        self.df = pd.read_csv(csv_file)

        #sort data by data
        self.df = self.df.sort_values(by=['Date'])

        self.balance = 1000.0 # initial balance of $1000
        self.stocks = 0.0 # initial number of stocks
        self.fees = 0.001 # 0.1% trading fees
        self.index = 0 # current index in the data frame

    def get_exchange_info(self):
        # get information about the exchange, such as supported symbols and trading fees
        pass

    def get_balance(self, symbol):
        # get the balance of a specific symbol in the user's account
        return self.balance

    def execute_order(self, order):
        # execute an order on the exchange
        ticker_data = self.df.iloc[self.index]
        current_price = ticker_data["Close"]

        if order["type"] == "BUY":
            if self.balance == 0.0:
                return
            
            # subtract trading fees from the balance
            balance = self.balance - (self.balance * self.fees)
            # calculate the amount of stocks to buy
            amount = balance / current_price

            print(f"Buying {amount} stocks for {balance}")

            self.balance = 0.0
            self.stocks = amount

        elif order["type"] == "SELL":
            if self.stocks == 0.0:
                return

            # calculate the balance for the amount of stocks
            balance = self.stocks * current_price
            # subtract trading fees from the balance
            balance = balance - (balance * self.fees)

            print(f"Selling {self.stocks} stocks for {balance}")

            self.balance = balance
            self.stocks = 0.0

    def cancel_order(self, order_id):
        # cancel an order on the exchange
        pass

    def get_ticker(self, symbol):
        # get ticker data for given symbol from exchange
        ticker_data = self.df.iloc[self.index]
        ticker = Ticker(symbol, ticker_data["Date"], ticker_data["Close"])
        self.index += 1
        return ticker


class MACDTradingAlgorithm:
    def __init__(self):
        self.data = []
        self.df = pd.DataFrame()

    def add_ticker(self, ticker):
        self.data.append(ticker)
        self.df = self.df.append({"Close": ticker.close, "Date": ticker.date, "Signal": ""}, ignore_index=True)

    def calculate(self, ticker, order_book):
        self.data.append(ticker)
        #self.df = self.df.append({"Close": ticker.close, "Date": ticker.date}, ignore_index=True)
        self.add_ticker(ticker)
        #print(self.df)

        # get index of last row
        i = len(self.df) - 1

        # Calculate the MACD
        macd = ta.trend.MACD(self.df["Close"].iloc[i-50:i+1])

        self.df["macd"] = macd.macd()
        self.df["macd_signal"] = macd.macd_signal()
        self.df["macd_diff"] = macd.macd_diff()

        signal, index = self.calculate_signal()

        #self.df["Signal"][index] = signal
        self.df.loc[index,"Signal"] = signal

        print(self.df)
        return signal
    
    def calculate_signal(self):
        #get index of last row
        i = len(self.df) - 1
        df = self.df
        if df['macd_signal'][i] > df['macd'][i] and df['macd_signal'][i-1] <= df['macd'][i-1]:
            # generate buy signal
            signal = "BUY"
            position = 1
        elif df['macd_signal'][i] < df['macd'][i] and df['macd_signal'][i-1] >= df['macd'][i-1]:
            # generate sell signal
            signal = "SELL"
            position = 0
        else:
            # no signal
            signal = "None"

        return signal, i


class Ticker:
    def __init__(self, symbol, date, close):
        self.symbol = symbol
        self.date = date
        self.close = close

if __name__ == "__main__":
    # create a trading interface for a specific exchange
    exchange = CsvExchange("ETHUSDT_1h_exchange", "./data2/Binance_ETHUSDT_1h.csv")
    trading_interface = TradingInterface(exchange)
    trading_algorithm = MACDTradingAlgorithm()

    # create a trading engine with the trading interface
    trading_engine = TradingEngine(trading_interface, trading_algorithm)

    # run the trading engine with the trading algorithm
    trading_engine.run()
