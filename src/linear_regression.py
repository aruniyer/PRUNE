import numpy as np
import tensorflow as tf
from basic_nn import Linear

def mse(y_pred, y_true):
    return tf.reduce_mean((y_pred - y_true)**2)

if __name__ == "__main__":
    f = np.asarray([[1, 1],[2, 1], [3, 1], [4, 1], [5, 1]], dtype=float)
    t = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=float)
    x = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 2])
    l1 = Linear(2, 2, "LinearRegression")
    cost = mse(l1(x), y)
    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {
            x: f,
            y: t
        }
        for i in range(100):
            sess.run(optimizer, feed_dict=feed_dict)
            print(sess.run(cost, feed_dict=feed_dict))