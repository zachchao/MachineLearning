import pickle 
from binance.client import Client


secret = open("secret.txt", "r").read()
api_key = secret.split(",")[0] 
api_secret = secret.split(",")[1]
client = Client(api_key, api_secret)


def getTradingSymbols():
    return client.get_all_tickers()


allSymbols = getTradingSymbols()[:30]
allSymbols = [symbol["symbol"] for symbol in allSymbols]
print(allSymbols)

totalTrainVals = [0, 0]
totalTestVals =[0, 0]

for symbol in allSymbols:
	trainVals = [0, 0]
	testVals = [0, 0]

	train_x, train_y, test_x, test_y = pickle.load(open('Pickles/{}_0.35_120.pickle'.format(symbol), 'rb'))

	
	for val in train_y:
		trainVals[0] += val[0]
		trainVals[1] += val[1]
		totalTrainVals[0] += val[0]
		totalTrainVals[1] += val[1]

	
	for val in test_y:
		testVals[0] += val[0]
		testVals[1] += val[1]
		totalTestVals[0] += val[0]
		totalTestVals[1] += val[1]

	print(symbol)
	print(trainVals, trainVals[0] / (trainVals[0] + trainVals[1]))
	print(testVals, testVals[0] / (testVals[0] + testVals[1]))

print(totalTrainVals, totalTrainVals[0] / (totalTrainVals[0] + totalTrainVals[1]))
print(totalTestVals, totalTestVals[0] / (totalTestVals[0] + totalTestVals[1]))
