import numpy as np
import tensorflow as tf
from basic_prune import PRUNE, calc_pmi, full_loss

if __name__ == "__main__":
    # parameters
    input_graph = "sample/graph.edgelist"
    embedding_size = 100
    hidden_size = 64
    num_epochs = 100
    lamb = 0.01
    learning_rate = 0.01
    
    graph = np.loadtxt(input_graph).astype(np.int32)
    nodeCount = graph.max() + 1
    out_degrees = np.zeros([nodeCount, 1])
    in_degrees = np.zeros([nodeCount, 1])
    for node_i, node_j in graph:
        out_degrees[node_i] += 1
        in_degrees[node_j] += 1
    pmis = calc_pmi(graph, in_degrees, out_degrees)
    out_degrees[out_degrees == 0] = 1
    in_degrees[in_degrees == 0] = 1
    
    pmi = tf.placeholder("float", [None, 1])
    source = tf.placeholder(tf.int32, [None])
    target = tf.placeholder(tf.int32, [None])
    outdeg = tf.placeholder("float", [None, 1])
    indeg = tf.placeholder("float", [None, 1])
    model = PRUNE(nodeCount, embedding_size, hidden_size, "PRUNE")
    cost = full_loss(model, hidden_size, source, target, indeg, outdeg, pmi, lamb)
    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {
            pmi: pmis,
            source: graph[:, 0],
            target: graph[:, 1],
            indeg: in_degrees[graph[:, 1]],
            outdeg: out_degrees[graph[:, 0]]
        }
        for i in range(num_epochs):
            sess.run(optimizer, feed_dict=feed_dict)
            print(sess.run(cost, feed_dict=feed_dict))