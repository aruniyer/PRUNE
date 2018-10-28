# PRUNE Restructured

import argparse
import sys
import numpy as np
import tensorflow as tf
from basic_prune import PRUNE, calc_pmi, full_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Restructured PRUNE.")

    parser.add_argument('--inputgraph', nargs='?',
                        default='sample/graph.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='graph.embeddings',
                        help='Output node embeddings of the graph')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Embedding dimension. Default is 128.')

    parser.add_argument('--lamb', type=float, default=0.01,
                        help='Parameter lambda in objective. Default is 0.01.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for Adam. Default is 1e-4.')

    parser.add_argument('--epoch', type=int, default=50,
                        help='Training epochs. Default is 50.')

    args = parser.parse_args()

    # parameters
    #input_graph = sys.argv[1]
    embedding_size = args.dimension
    hidden_size = 128
    latent_size = 64
    num_epochs = args.epoch
    lamb = args.lamb
    learning_rate = args.learning_rate
    
    graph = np.loadtxt(args.inputgraph).astype(np.int32)
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
    model = PRUNE(nodeCount, embedding_size, hidden_size, latent_size, "PRUNE")
    cost = full_loss(model, latent_size, source, target, indeg, outdeg, pmi, lamb)
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
            print("Epoch-->" + str(i+1) + " ...")
            sess.run(optimizer, feed_dict=feed_dict)
            print(sess.run(cost, feed_dict=feed_dict))
            if(i % 10 == 0):
                embs = sess.run(model.E.W)
                filename = "re_embeddings_epoch" + str(i)
                np.savetxt(filename, embs, delimiter = ",")
        # final_embeddings = sess.run(model.E.W)
        # np.savetxt(args.output, final_embeddings, delimiter = ",")