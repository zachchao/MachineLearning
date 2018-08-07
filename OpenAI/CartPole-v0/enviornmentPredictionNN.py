import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')

# dataSize is how much data we want
def generateTrainingData(dataSize):
    logits = np.zeros(shape=(dataSize, 5))
    labels = np.zeros(shape=(dataSize, 4))
    index = 0
    episodes= 0
    while index != dataSize:
        observation = env.reset()
        for t in range(200):
            action = env.action_space.sample()
            # I feel like itll be a lot easier for it if its -1
            if action == 0:
                action = -1
            # Add the data
            logit = np.append(observation, action)
            logits[index] = logit
            if action == -1:
                action = 0
            observation, reward, done, info = env.step(action)
            labels[index] = observation
            index += 1
            if done or index == dataSize:
                break
        episodes += 1
    print(episodes)
    return logits, labels

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(history.epoch, np.array(history.history['acc']), 
        label='Accuracy')
    plt.plot(history.epoch, np.array(history.history['loss']), 
        label='Loss')
    plt.legend()
    plt.ylim([0,1.5])
    plt.show()



trainLogits, trainLabels = generateTrainingData(2000)
testLogits, testLabels = generateTrainingData(1000)

# Normalize
mean = trainLogits.mean(axis=0)
std = trainLogits.std(axis=0)
mean[4] = 0
std[4] = 1
trainLogits = (trainLogits - mean) / std
trainLabels = (trainLabels - mean[:4]) / std[:4]
testLogits = (testLogits - mean) / std
testLabels = (testLabels - mean[:4]) / std[:4]

model = keras.Sequential([
    keras.layers.Dense(10, activation=tf.nn.relu,
        input_shape=(trainLogits.shape[1],)),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(40, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu)
])
model.compile(loss='mse',
    optimizer=tf.train.RMSPropOptimizer(0.001),
    metrics=['accuracy'])
history = model.fit(trainLogits, trainLabels, epochs=500,
    validation_split=0.2, verbose=0)

#plot_history(history)

test_loss, test_acc = model.evaluate(testLogits, testLabels)
print('Test accuracy:', test_acc)

def determine_action(observation):
    #print(observation)
    toPredict = np.zeros(shape=(2, 5))
    toPredict[0] = np.append(observation, -1)
    toPredict[1] = np.append(observation, 1)
    test_predictions = model.predict(toPredict)
    #print(test_predictions)
    if abs(test_predictions[0][2]) < abs(observation[2]):
        return 0
    return 1

for i in range(20):
    observation = env.reset()
    for t in range(200):
        action = determine_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print(t)
            break
