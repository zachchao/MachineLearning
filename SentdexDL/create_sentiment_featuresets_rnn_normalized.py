from binance.client import Client
import pandas as pd
from stockstats import StockDataFrame as Sdf
from time import time
import numpy as np
import json
import collections
import pickle


secret = open("secret.txt", "r").read()
api_key = secret.split(",")[0] 
api_secret = secret.split(",")[1]
client = Client(api_key, api_secret)


def getTradingSymbols():
    return client.get_all_tickers()


def getKlineData(symbol, interval, startTime=None, endTime=None):
    try:
        return client.get_klines(symbol=symbol, interval=interval, startTime=startTime, endTime=endTime)
    except json.decoder.JSONDecodeError:
        return getKlineData(symbol, interval)


def getKlines(symbol, interval, startTime=None, endTime=None):
    klineData = getKlineData(symbol, interval, startTime, endTime)
    klineData = {
        'time' : [kline[0] for kline in klineData],
        'open' : [float(kline[1]) for kline in klineData],
        'high' : [float(kline[2]) for kline in klineData],
        'low' : [float(kline[3]) for kline in klineData],
        'close' : [float(kline[4]) for kline in klineData]
    }
    return klineData


def getDayOfKlines1Min(symbol, startTime):
    interval = Client.KLINE_INTERVAL_1MINUTE
    # Sends 500 klines, each worth 1 minute
    klines = []
    # Hours to minutes to seconds
    klineRange = 8 * 60 * 60
    for i in range(2, -1, -1):
        klines += getKlineData(
            symbol=symbol, 
            interval=interval, 
            # Convert to milliseconds
            startTime=str(int((startTime + (i * klineRange)) * 1000)), 
            endTime=str(int((startTime + ((i + 1) * klineRange)) * 1000))
        )

    klineData = {
        'time': [kline[0] for kline in klines],
        'open': [float(kline[1]) for kline in klines],
        'close': [float(kline[4]) for kline in klines],
        'high': [float(kline[2]) for kline in klines],
        'low': [float(kline[3]) for kline in klines]
    }

    return klineData


def getAllTA(symbol, startTime):
    #tickers = getKlines(symbol, interval)
    windowSize = 200
    tickers = getDayOfKlines1Min(symbol, startTime)

    labels = []
    logits = []

    df = pd.DataFrame(tickers)
    stock_df = Sdf.retype(df)
    
    '''
    TAOptions = ['open_2_d', 'open_-2_r', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 
        'open_2_sma', 'macd', 'macds', 'macdh', 'boll', 'boll_ub', 'boll_lb', 
        'close_10.0_le_5_c', 'cr-ma2_xu_cr-ma1_20_c', 'rsi_6', 'rsi_12', 
        'wr_10', 'wr_6', 'cci', 'cci_20', 'tr', 'atr', 'dma', 'pdi', 'mdi', 'dx', 
        'adx', 'adxr', 'trix', 'trix_9_sma']
    '''
    TAOptions = []
    for TA in TAOptions:
        stock_df.get(TA)

    del stock_df['time']

    df_values = stock_df.values
    #df_values = tickers

    for i in range(windowSize + 15, len(df_values)):
        indivList = []
        p0 = df_values[i - windowSize][0]
        for indivTicker in list(df_values[i - windowSize : i]):
            indivTicker = [i / p0 for i in indivTicker]
            indivList += list(indivTicker)
        logits.append(indivList)

        # Rise in the close price
        if df_values[i - 1][1] < df_values[i][1]:
            labels.append([1, 0])
        else:
            labels.append([0, 1])

    return list(logits), labels


def hasExistedForAtLeast(symbol, days):
    return len(client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY , str(days) + " day ago UTC")) == days


if __name__ == '__main__':
    symbols = getTradingSymbols()
    testSymbol = symbols[0]["symbol"]

    test_day_range = 3
    test_hour_range = 8

    # Last test_hour_range hours in seconds
    test_start_time = time() - (test_hour_range * 60 * 60)
    test_x, test_y = getAllTA(testSymbol, test_start_time)


    # The past x test_day_range excluding last test_hour_range hours
    train_x, train_y = getAllTA(testSymbol, test_start_time - ((test_day_range + 1) * 86400))

    for symbol in symbols[:1]:
        if hasExistedForAtLeast(symbol["symbol"], test_day_range):
            print("Generating data for ", symbol["symbol"])
            for i in range(test_day_range, 0, -1):
                temp_train_x, temp_train_y = getAllTA(symbol["symbol"], test_start_time - (test_day_range * 86400))

                train_x += temp_train_x
                train_y += temp_train_y
        else:
            print(symbol["symbol"], " has not existed for long enough")


    with open("rnn_sentiment_set_normalized4.pickle", 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
