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


rises = 0
falls = 0
difference = 0
count = 0
overAllDifference = 0
overAllCount = 0

# Sliding window size for labels
labelWindowSize = 120

minimumPercentageRise = 0.35


def getTradingSymbols():
    return client.get_all_tickers()


def getKlineData(symbol, interval, startTime=None, endTime=None):
    try:
        return client.get_klines(symbol=symbol, interval=interval, startTime=startTime, endTime=endTime)
    except json.decoder.JSONDecodeError:
        return getKlineData(symbol, interval)


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
    global rises, falls, difference, count, overAllDifference, overAllCount, labelWindowSize, minimumPercentageRise
    # Sliding window size for features
    featureWindowSize = 600

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

    # Labels will be a sliding window on the preceeding 30 minutes
    # Keeping the highest closing price in order to see if there can
    # be an expected rise within the next half hour 
    #labelWindow = df_values[featureWindowSize : featureWindowSize + labelWindowSize]
    #labelWindow = [i[1] for i in labelWindow]
    #highestFuture = max(labelWindow)

    for i in range(len(df_values) - featureWindowSize - labelWindowSize):
        labelWindow = df_values[featureWindowSize + i : featureWindowSize + labelWindowSize + i]
        labelWindow = [i[1] for i in labelWindow]
        highestFuture = max(labelWindow)

        normalizedList = []
        # Normalize the data
        featureWindow = list(df_values[i : featureWindowSize + i])

        # The open price of this featureWindow
        p0 = featureWindow[0][0]
        
        for ticker in featureWindow:
            indivTicker = [i / p0 for i in ticker]
            normalizedList += list(indivTicker)
        logits.append(normalizedList)

        # Rise in the close price in the next half hour
        # A rise beyond minimumPercentageRise
        riseMinimum = minimumPercentageRise / 100

        # Iterate through, if we hit our goal rise percentage we give [1, 0]
        # If we hit a loss of riseMinimum / 2 first, [0, 1]
        # If we hit neither, count that as a negative as well, [0, 1]

        found = False
        currentMinute = df_values[featureWindowSize + i][1]
        for ticker in labelWindow:
            if ticker > currentMinute * (1 + riseMinimum):        
                labels.append([1, 0])
                rises += 1
                found = True
                break
        if not found:
            labels.append([0, 1])
            falls += 1 

            difference += (df_values[featureWindowSize + labelWindowSize + i][1] / df_values[featureWindowSize + i][1])
            count += 1
            overAllDifference += (df_values[featureWindowSize + labelWindowSize + i][1] / df_values[featureWindowSize + i][1])
            overAllCount += 1

        

        # Simulate the sliding window by using a new highest
        #highestFuture = max(highestFuture, df_values[i + featureWindowSize + labelWindowSize][1])
        

    return list(logits), labels


def hasExistedForAtLeast(symbol, days):
    return len(client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY , str(days) + " day ago UTC")) == days


if __name__ == '__main__':
    symbols = getTradingSymbols()
    allSymbols = []
    test_day_range = 20
    test_hour_range = 48

    for symbol in symbols[ : 30]:
        symbol = symbol["symbol"]
        if hasExistedForAtLeast(symbol, 30):
            rises = 0
            falls = 0
            difference = 0
            count = 0

            print("Generating data for", symbol)
            # Last test_hour_range hours in seconds
            test_start_time = time() - (test_hour_range * 60 * 60)
            test_x, test_y = getAllTA(symbol, test_start_time)

            # The past x test_day_range excluding last test_hour_range hours
            train_x, train_y = getAllTA(symbol, test_start_time - ((test_day_range + 1) * 86400))

            for i in range(test_day_range, 0, -1):
                
                temp_train_x, temp_train_y = getAllTA(symbol, test_start_time - (test_day_range * 86400))

                train_x += temp_train_x
                train_y += temp_train_y

            print(rises / (rises +  falls) * 100, " percent rose")
            print(falls / (rises +  falls) * 100 , " percent fell")
            print("Average difference after {} minutes ".format(labelWindowSize), difference / count)
            print("Overall average difference after {} minutes ".format(labelWindowSize), overAllDifference / overAllCount)
            print(len(train_x), len(test_x))

            with open("Pickles/{}_{}_{}.pickle".format(symbol, minimumPercentageRise, labelWindowSize), 'wb') as f:
                pickle.dump([train_x, train_y, test_x, test_y], f)
            allSymbols += symbol
            print("")

    print(allSymbols)
