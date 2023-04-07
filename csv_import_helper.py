import numpy as np
import pandas as pd
import os

def import_csv_from_cryptodatadownload(csv_file, symbol):
    df = pd.read_csv(csv_file)

    #correct date format
    df['date'] = pd.to_datetime(df['date'])
    
    #extract symbol name from data
  #  symbol = df.iat[0,2]
  #  print(symbol)
   # index = symbol.find('/')
  #  symbol = symbol[0:index]
  #  print(symbol)

    #drop columns and prepare volume column
    df = df.drop(columns=['symbol','Volume USDT','tradecount'])
    column_name = "Volume {0}".format(symbol)
    df.rename(columns={column_name: 'volume'}, inplace=True)

    #sort entries by date to make sure oldest entries come first
    df.sort_values('date', ascending=True, inplace=True)
    df.set_index('date', inplace=True)
    return df

def import_example_sin_data(length):
    in_array = np.linspace(0, 150, length)
    out_array = (np.sin(in_array)*40) + 100

    df = pd.DataFrame(data=out_array, columns=["price"])
    # df['open'] = df['close']
    # df['high'] = df['close']
    # df['low'] = df['close']
    # df['volume'] = 1000.0

    return df
