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
    klineRange = 480 * 60
    for i in range(3):
        klines += getKlineData(
            symbol=symbol, 
            interval=interval, 
            startTime=str(round((startTime + (i * klineRange)) * 1000)), 
            endTime=str(round((startTime + ((i + 1) * klineRange)) * 1000))
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
    tickers = getDayOfKlines1Min(symbol, startTime)

    labels = []
    logits = []

    df = pd.DataFrame(tickers)
    stock_df = Sdf.retype(df)
    TAOptions = ['open_2_d', 'open_-2_r', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 
        'open_2_sma', 'macd', 'macds', 'macdh', 'boll', 'boll_ub', 'boll_lb', 
        'close_10.0_le_5_c', 'cr-ma2_xu_cr-ma1_20_c', 'rsi_6', 'rsi_12', 
        'wr_10', 'wr_6', 'cci', 'cci_20', 'tr', 'atr', 'dma', 'pdi', 'mdi', 'dx', 
        'adx', 'adxr', 'trix', 'trix_9_sma']

    for TA in TAOptions:
        stock_df.get(TA)

    del stock_df['time']


    df_values = stock_df.values

    for i in range(100 + 15, len(df_values)):
        logits.append(df_values[i - 100 : i])
        # Rise in the close price
        if df_values[i - 1][1] < df_values[i][1]:
            labels.append([1, 0])
        else:
            labels.append([0, 1])

    return list(logits), labels


def hasExistedForAtLeast(symbol, days):
    return len(client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1DAY , str(days) + " day ago UTC")) == days


if __name__ == '__main__':
    symbols = getTradingSymbols()
    symbol = symbols[0]["symbol"]

    days = 10

    # The past x days excluding today
    train_x, train_y = getAllTA(symbol, time() - ((days + 1) * 86400))

    for i in range(days):
        if hasExistedForAtLeast(symbol, days + 1):
            temp_train_x, temp_train_y = getAllTA(symbol, time() - ((days + 1) * 86400))

            train_x += temp_train_x
            train_y += temp_train_y


    # Today
    test_x, test_y = getAllTA(symbol, time() - (1 * 10000))
    
    if hasExistedForAtLeast(symbol, days + 1):
        temp_test_x, temp_test_y = getAllTA(symbol, time() - (1 * 10000))

        test_x += temp_test_x
        test_y += temp_test_y


    with open("rnn_sentiment_set.pickle", 'wb') as f:
        pickle.dump([train_x[15:], train_y[15:], test_x[15:], test_y[15:]], f)
