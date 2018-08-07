import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pickle
import numpy as np
import random


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Shuffles two arrays and keeps them parllel
def shuffle(logits, labels):
    randomIndices = [i for i in range(len(labels))]
    random.shuffle(randomIndices)

    newLogits = [None for i in range(len(logits))]
    newLabels = [None for i in range(len(labels))]
    for i in range(len(randomIndices)):
        newLogits[i] = logits[randomIndices[i]]
        newLabels[i] = labels[randomIndices[i]]
    return newLogits, newLabels


def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size) 
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output


def train_neural_network(x):
    global test_x, test_y
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for symbol in symbols:
            print("\nSymbol : ", symbol)
            train_x, train_y, temp_test_x, temp_test_y = pickle.load(open("Pickles/{}_0.35_120.pickle".format(symbol).format(symbol), 'rb'))
            
            train_x = train_x[ : len(train_x) - (len(train_x) % batch_size)]
            train_y = train_y[ : len(train_y) - (len(train_y) % batch_size)]

            test_x += temp_test_x
            test_y += temp_test_y

            print("Test set now has ", len(test_y), " sets")

            for epoch in range(hm_epochs):
                epoch_loss = 0

                train_x, train_y = shuffle(train_x, train_y)
                for i in range(0, len(train_x), batch_size):
                    batch_x = np.array(train_x[i : i + batch_size])
                    batch_y = np.array(train_y[i : i + batch_size])
 
                    batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))

                    _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss)


            test_x = test_x[ : len(test_x) - (len(test_x) % batch_size)]
            test_y = test_y[ : len(test_y) - (len(test_y) % batch_size)]
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x: np.array(test_x).reshape((-1, n_chunks, chunk_size)), y: test_y}))

        
        test_x = test_x[ : len(test_x) - (len(test_x) % batch_size)]
        test_y = test_y[ : len(test_y) - (len(test_y) % batch_size)]
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x: np.array(test_x).reshape((-1, n_chunks, chunk_size)), y: test_y}))

        print("Saving")
        saver = tf.train.Saver()
        saver.save(sess, "Models/model_1.0_120.ckpt")


if __name__ == "__main__":
    symbols = ['ETHBTC', 'LTCBTC', 'BNBBTC', 'NEOBTC', 'QTUMETH', 'EOSETH', 'SNTETH', 'BNTETH', 'BCCBTC', 'GASBTC', 'BNBETH', 'BTCUSDT', 'ETHUSDT', 'HSRBTC', 'OAXETH', 'DNTETH', 'MCOETH', 'ICNETH', 'MCOBTC', 'WTCBTC', 'WTCETH', 'LRCBTC', 'LRCETH', 'QTUMBTC', 'YOYOBTC', 'OMGBTC', 'OMGETH', 'ZRXBTC', 'ZRXETH', 'STRATBTC']
    symbols = symbols[ : 30]

    train_x, train_y, test_x, test_y = pickle.load(open("Pickles/{}_0.35_120.pickle".format(symbols[0]), 'rb'))

    hm_epochs = 3
    n_classes = 2
    batch_size = 100
    chunk_size = 4
    n_chunks = 600
    rnn_size = 600

    x = tf.placeholder('float', [None, n_chunks, chunk_size])
    y = tf.placeholder('float')

    print("RNN HAS BEGUN LEARNING OH GOD")
    train_neural_network(x)