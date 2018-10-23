import numpy as np
import tensorflow as tf
from basic_prune import PRUNE, calc_pmi, proximity_loss

def ranking_loss(model, transition_matrix):
    all_ranks = model.all_rank()
    v1 = tf.sparse_tensor_dense_matmul(sp_a=transition_matrix, b=all_ranks)
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
    out_degrees = np.zeros([nodeCount, 1])
    in_degrees = np.zeros([nodeCount, 1])
    for node_i, node_j in graph:
        out_degrees[node_i] += 1
        in_degrees[node_j] += 1
    
    data = []
    rows = []
    cols = []
    for node_i, node_j in graph:
        if (out_degrees[node_j] > 0):
            rows.append(node_i)
            cols.append(node_j)
            data.append(1.0 / float(out_degrees[node_j]))
    data = np.array(data, copy=False)
    rows = np.array(rows, dtype=np.int32, copy=False)
    cols = np.array(cols, dtype=np.int32, copy=False)
    sparse_trans_mat = tf.SparseTensorValue(indices=np.array([rows, cols]).T, values=data, dense_shape=[nodeCount, nodeCount])
    pmis = calc_pmi(graph, in_degrees, out_degrees)
    
    pmi = tf.placeholder("float", [M, 1])
    source = tf.placeholder(tf.int32, [M])
    target = tf.placeholder(tf.int32, [M])
    transition_matrix = tf.sparse_placeholder(tf.float32)
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
            transition_matrix: sparse_trans_mat
        }
        for i in range(num_epochs):
            sess.run(optimizer, feed_dict=feed_dict)
            print(sess.run(cost, feed_dict=feed_dict))