import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import os
import pickle
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train_x, train_y, test_x, test_y = pickle.load(open('sentiment_set2.pickle', 'rb'))
train_x = train_x.values[15:]
train_y = train_y[15:]
test_x = test_x.values[15:]
test_y = test_y[15:]

del train_x['high']
del train_x['low']
del train_x['open']
del train_x['close']
del train_x['high']
#mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 100
n_nodes_hl2 = 500
n_nodes_hl3 = 250
n_nodes_hl4 = 10

n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl4]))}
                      
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                      'biases' : tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.add(tf.matmul(l4, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for i in range(0, len(train_x), batch_size):
				start = i
				end = i + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
			print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 0), tf.argmax(y, 0))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x: test_x, y: test_y}))


if __name__ == "__main__":
  train_neural_network(x)
