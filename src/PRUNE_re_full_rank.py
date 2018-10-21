import numpy as np
import tensorflow as tf

class Linear:
    def __init__(self, input_size, output_size, scope):
        self.input_size = input_size
        self.output_size = output_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.W = tf.get_variable("W_linear", [input_size, output_size])
            self.b = tf.get_variable("b_linear", [1, output_size])
    
    def __call__(self, x):
        return tf.add(tf.matmul(x, self.W), self.b)

class Embedding:
    def __init__(self, embedding_size, instances, scope):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.W = tf.get_variable("Embeddings", [instances, embedding_size])
            tf.summary.scalar('mean', tf.reduce_mean(self.W))
    
    def __call__(self, ind):
        return tf.gather(self.W, ind)

class Proximity:
    def __init__(self, embedding_size, hidden_size, scope):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.l1 = Linear(embedding_size, hidden_size, "LinearLayer1")
            self.l2 = Linear(hidden_size, hidden_size, "LinearLayer2")
    
    def __call__(self, x):
        return tf.nn.relu(self.l2(tf.nn.elu(self.l1(x))))

class Ranking:
    def __init__(self, embedding_size, hidden_size, scope):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.l1 = Linear(embedding_size, hidden_size, "LinearLayer1")
            self.l2 = Linear(hidden_size, 1, "LinearLayer2")
    
    def __call__(self, x):
        return tf.nn.softplus(self.l2(tf.nn.elu(self.l1(x))))

class PRUNE:
    def __init__(self, instances, embedding_size, hidden_size, scope):
        self.instances = instances
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.E = Embedding(embedding_size=embedding_size, instances=instances, scope="NodeEmbedding")
            self.P = Proximity(embedding_size=embedding_size, hidden_size=hidden_size, scope="ProximityLayer")
            self.R = Ranking(embedding_size=embedding_size, hidden_size=hidden_size, scope="RankingLayer")

    def proximity_representation(self, ind):
        return self.P(self.E(ind))
    
    def rank(self, ind):
        return self.R(self.E(ind))
    
    def all_rank(self):
        return self.R(self.E(range(self.instances)))

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
    z_i = model.proximity_representation(source)
    z_j = model.proximity_representation(target)
    zWz = z_i * tf.matmul(z_j, W_shared_posi)
    zWz = tf.reduce_sum(zWz, 1, keep_dims=True)
    return tf.reduce_mean((zWz - pmis)**2)

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
            if (out_degrees[i] != 0):
                trans_mat[i, j] = 1.0 / out_degrees[i]
    pmis = calc_pmi(graph, in_degrees, out_degrees)
    out_degrees[out_degrees == 0] = 1
    in_degrees[in_degrees == 0] = 1
    
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