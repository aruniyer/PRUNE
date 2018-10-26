import numpy as np
import tensorflow as tf
from basic_nn import Linear, Embedding

class Proximity:
    def __init__(self, embedding_size, hidden_size, latent_size, scope):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.l1 = Linear(embedding_size, hidden_size, "LinearLayer1")
            self.l2 = Linear(hidden_size, latent_size, "LinearLayer2")
    
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
    def __init__(self, instances, embedding_size, hidden_size, latent_size, scope):
        self.instances = instances
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.scope = scope
        with tf.variable_scope(scope):
            self.E = Embedding(embedding_size=embedding_size, instances=instances, scope="NodeEmbedding")
            self.P = Proximity(embedding_size=embedding_size, hidden_size=hidden_size, latent_size=latent_size, scope="ProximityLayer")
            self.R = Ranking(embedding_size=embedding_size, hidden_size=hidden_size, scope="RankingLayer")

    def proximity_representation(self, ind):
        return self.P(self.E(ind))
    
    def rank(self, ind):
        return self.R(self.E(ind))
    
    def all_rank(self):
        return self.rank(range(self.instances))

def calc_pmi(graph, in_degrees, out_degrees, alpha=5.0):
    pmis = np.zeros((len(graph), 1))
    for ind in range(len(graph)):
        head, tail = graph[ind]
        pmi = len(graph) / alpha / out_degrees[head] / in_degrees[tail]
        pmis[ind, 0] = np.log(pmi)

    pmis[pmis < 0] = 0

    return pmis

def proximity_loss(model, latent_size, source, target, pmis):
    W_init = np.identity(latent_size)
    W_init += abs(np.random.randn(latent_size, latent_size) / 1000.0)
    W_initializer = tf.constant_initializer(W_init)
    W_shared = tf.get_variable("W_shared", [latent_size, latent_size],
                                initializer=W_initializer)
    W_shared_posi = tf.nn.relu(W_shared)
    z_i = model.proximity_representation(source)
    z_j = model.proximity_representation(target)
    zWz = z_i * tf.matmul(z_j, W_shared_posi)
    zWz = tf.reduce_sum(zWz, 1, keep_dims=True)
    return tf.reduce_mean((zWz - pmis)**2)

def ranking_loss(model, source, target, indeg, outdeg):
    heads_pi = model.rank(source)
    tails_pi = model.rank(target)
    r_loss = indeg * (tf.square(-tails_pi / indeg + heads_pi / outdeg))
    return tf.reduce_mean(r_loss)

def full_loss(model, hidden_size, source, target, indeg, outdeg, pmis, lamb):
    p_loss = proximity_loss(model, hidden_size, source, target, pmis)
    r_loss = ranking_loss(model, source, target, indeg, outdeg)
    return p_loss + lamb * r_loss