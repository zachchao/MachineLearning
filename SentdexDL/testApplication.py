import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pickle
import numpy as np
import random


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def one_hot(x):
    return [int(i // max(x)) for i in x]


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


def run_neural_network(x):
    global correctBuys, avoidedBuys, correctAvoid, incorrectBuys

    prediction = recurrent_neural_network(x)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, "Models/model_15.ckpt")
        print("Model restored.")


        symbols = ['ETHBTC', 'LTCBTC', 'BNBBTC', 'NEOBTC', 'QTUMETH', 'EOSETH', 'SNTETH', 'BNTETH', 'BCCBTC', 'GASBTC', 'BNBETH', 'BTCUSDT', 'ETHUSDT', 'HSRBTC', 'OAXETH', 'DNTETH', 'MCOETH', 'ICNETH', 'MCOBTC', 'WTCBTC', 'WTCETH', 'LRCBTC', 'LRCETH', 'QTUMBTC', 'YOYOBTC', 'OMGBTC', 'OMGETH', 'ZRXBTC', 'ZRXETH', 'STRATBTC']
        symbols = symbols[ : 30]
        for symbol in symbols:
            train_x, train_y, test_x, test_y = pickle.load(open('Pickles/{}_0.35_120.pickle'.format(symbol), 'rb'))
            print("Loaded data for ", symbol)
            for i in range(len(test_x)):
                batch_x = np.array(test_x[i])
                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))
                actual = test_y[i]
                #print("Loaded test data, actual is ", actual)

                out = sess.run(prediction, feed_dict={x : batch_x})[0]
                one_hot_ray = one_hot(out)

                # Should have bought
                if test_y[i] == [1, 0]:
                    # Correctly bought
                    if one_hot_ray == [1, 0]:
                        correctBuys += 1
                    else:
                        avoidedBuys += 1
                else:
                    if one_hot_ray == [0, 1]:
                        correctAvoid += 1
                    else:
                        incorrectBuys += 1

            print("Correct buys : ", correctBuys)
            print("Avoided buys : ", avoidedBuys)
            print("Correctly avoided : ", correctAvoid)
            print("Incorrect buys : ", incorrectBuys)



if __name__ == "__main__":
    hm_epochs = 2
    n_classes = 2
    batch_size = 1
    chunk_size = 4
    n_chunks = 600
    rnn_size = 600

    correctBuys = 0
    avoidedBuys = 0
    correctAvoid = 0
    incorrectBuys = 0


    x = tf.placeholder('float', [None, n_chunks, chunk_size])
    y = tf.placeholder('float')

    run_neural_network(x)
