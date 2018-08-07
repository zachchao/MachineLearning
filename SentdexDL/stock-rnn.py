import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pickle
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train_x, train_y, test_x, test_y = pickle.load(open('Pickles/BNBBTC_15.pickle', 'rb'))

hm_epochs = 3
n_classes = 2
batch_size = 100
chunk_size = 4
n_chunks = 800
rnn_size = 800

train_x = train_x[ : len(train_x) - (len(train_x) % batch_size)]
train_y = train_y[ : len(train_y) - (len(train_y) % batch_size)]
test_x = test_x[ : len(test_x) - (len(test_x) % batch_size)]
test_y = test_y[ : len(test_y) - (len(test_y) % batch_size)]

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


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
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(0, len(train_x), batch_size):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x: np.array(test_x).reshape((-1, n_chunks, chunk_size)), y: test_y}))


if __name__ == "__main__":
    print("RNN HAS BEGUN LEARNING OH GOD")
    train_neural_network(x)