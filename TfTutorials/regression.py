import tensorflow as tf
from tensorflow import keras
import numpy as np


boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# Test data is *not* used when calculating the mean and std.

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


model = keras.Sequential([
	keras.layers.Dense(64, activation=tf.nn.relu, 
	    input_shape=(train_data.shape[1],)),
	keras.layers.Dense(64, activation=tf.nn.relu),
	keras.layers.Dense(1)
])

model.compile(loss='mse',
    optimizer=tf.train.RMSPropOptimizer(0.001),
    metrics=['mae'])

#model.summary()

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
	validation_split=0.2, verbose=0)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))
