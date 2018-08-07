import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

print(result)

with tf.Session() as sess:
	print(sess.run(result))

	