import numpy as np
import tensorflow as tf

class Linear:
    def __init__(self, input_size, output_size, scope):
        with tf.variable_scope(scope):
            self.W = tf.get_variable("W_linear", [input_size, output_size])
            self.b = tf.get_variable("b_linear", [1, output_size])
    
    def __call__(self, x):
        return tf.add(tf.matmul(x, self.W), self.b)

class Embedding:
    def __init__(self, embedding_size, instances, scope):
        with tf.variable_scope(scope):
            self.W = tf.get_variable("Embeddings", [instances, embedding_size])
            tf.summary.scalar('mean', tf.reduce_mean(self.W))
    
    def __call__(self, ind):
        return tf.gather(self.W, ind)

class Proximity:
    def __init__(self, embedding_size, hidden_size, scope):
        with tf.variable_scope(scope):
            self.l1 = Linear(embedding_size, hidden_size, "LinearLayer1")
            self.l2 = Linear(hidden_size, hidden_size, "LinearLayer2")
    
    def __call__(self, x):
        return tf.nn.relu(self.l2(tf.nn.elu(self.l1(x))))

class PRUNE:
    def __init__(self, instances, embedding_size, hidden_size):
        self.E = Embedding(embedding_size=embedding_size, instances=instances, scope="NodeEmbedding")
        self.P = Proximity(embedding_size=embedding_size, hidden_size=hidden_size, scope="ProximityLayer")

    def __call__(self, ind):
        return self.P(self.E(ind))

def calc_pmi(graph, in_degrees, out_degrees, alpha=5.0):
    pmis = np.zeros((len(graph), 1))
    for ind in range(len(graph)):
        head, tail = graph[ind]
        pmi = len(graph) / alpha / out_degrees[head] / in_degrees[tail]
        pmis[ind, 0] = np.log(pmi)

    pmis[pmis < 0] = 0

    return pmis

def proximity_loss(model, hidden_size, source, target, pmis):
    W_init = np.identity(hidden_size)
    W_init += abs(np.random.randn(hidden_size, hidden_size) / 1000.0)
    W_initializer = tf.constant_initializer(W_init)
    W_shared = tf.get_variable("W_shared", [hidden_size, hidden_size],
                                initializer=W_initializer)
    W_shared_posi = tf.nn.relu(W_shared)
    z_i = model(source)
    z_j = model(target)
    zWz = z_i * tf.matmul(z_j, W_shared_posi)
    zWz = tf.reduce_sum(zWz, 1, keep_dims=True)
    return tf.reduce_mean((zWz - pmis)**2)

if __name__ == "__main__":
    graph = np.loadtxt("sample/graph.edgelist").astype(np.int32)
    nodeCount = graph.max() + 1
    out_degrees = np.zeros(nodeCount)
    in_degrees = np.zeros(nodeCount)
    for node_i, node_j in graph:
        out_degrees[node_i] += 1
        in_degrees[node_j] += 1
    pmis = calc_pmi(graph, in_degrees, out_degrees)
    
    embedding_size = 100
    hidden_size = 64
    num_epochs = 100
    pmi = tf.placeholder("float", [None, 1])
    source = tf.placeholder(tf.int32, [None])
    target = tf.placeholder(tf.int32, [None])
    model = PRUNE(nodeCount, embedding_size, hidden_size)
    cost = proximity_loss(model, hidden_size, source, target, pmi)
    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {
            pmi: pmis,
            source: graph[:, 0],
            target: graph[:, 1]
        }
        for i in range(num_epochs):
            sess.run(optimizer, feed_dict=feed_dict)
            print(sess.run(cost, feed_dict=feed_dict))