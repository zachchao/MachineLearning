import pickle 
from binance.client import Client


secret = open("secret.txt", "r").read()
api_key = secret.split(",")[0] 
api_secret = secret.split(",")[1]
client = Client(api_key, api_secret)


def getTradingSymbols():
    return client.get_all_tickers()


symbols = getTradingSymbols()[:30]
symbols = [symbol["symbol"] for symbol in symbols]
symbol = symbols[0]

train_x, train_y, test_x, test_y = pickle.load(open('Pickles/{}_1.0_120.pickle'.format(symbol), 'rb'))

print(test_x[0])
print(test_y[0])
