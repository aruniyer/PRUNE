import numpy as np
import tensorflow as tf
from basic_prune import PRUNE, calc_pmi, proximity_loss

def ranking_loss(model, transition_matrix):
    all_ranks = model.all_rank()
    v1 = tf.reduce_sum(all_ranks * tf.transpose(transition_matrix), axis=1)
    return tf.reduce_mean(tf.square(v1 - all_ranks))

def full_loss(model, hidden_size, source, target, transition_matrix, pmis, lamb):
    p_loss = proximity_loss(model, hidden_size, source, target, pmis)
    r_loss = ranking_loss(model, transition_matrix)
    return p_loss + lamb * r_loss

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
    M = len(graph[:, 0])
    out_degrees = np.zeros(nodeCount)
    in_degrees = np.zeros(nodeCount)
    for node_i, node_j in graph:
        out_degrees[node_i] += 1
        in_degrees[node_j] += 1
    trans_mat = np.zeros([nodeCount, nodeCount])
    for i in range(nodeCount):
        for j in range(nodeCount):
            if (out_degrees[i] > 0):
                trans_mat[i, j] = 1.0 / out_degrees[i]
    pmis = calc_pmi(graph, in_degrees, out_degrees)
    
    pmi = tf.placeholder("float", [M, 1])
    source = tf.placeholder(tf.int32, [M])
    target = tf.placeholder(tf.int32, [M])
    transition_matrix = tf.placeholder("float", [nodeCount, nodeCount])
    model = PRUNE(nodeCount, embedding_size, hidden_size, "PRUNE")
    cost = full_loss(model, hidden_size, source, target, transition_matrix, pmi, lamb)
    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {
            pmi: pmis,
            source: graph[:, 0],
            target: graph[:, 1],
            transition_matrix: trans_mat
        }
        for i in range(num_epochs):
            sess.run(optimizer, feed_dict=feed_dict)
            print(sess.run(cost, feed_dict=feed_dict))