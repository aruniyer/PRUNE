import tensorflow as tf

class Linear:
    def __init__(self, input_size, output_size, scope, seed = 1.00):
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        self.scope = scope
        with tf.variable_scope(scope):
            self.W = tf.get_variable(name = "W_linear", shape = [input_size, output_size], initializer = tf.glorot_uniform_initializer(seed = self.seed))
            self.b = tf.get_variable(name = "b_linear", shape = [1, output_size], initializer = tf.glorot_uniform_initializer(seed = self.seed))
    
    def __call__(self, x):
        return tf.add(tf.matmul(x, self.W), self.b)

class Embedding:
    def __init__(self, embedding_size, instances, scope, seed = 1.00):
        self.embedding_size = embedding_size
        self.instances = instances
        self.seed = seed
        self.scope = scope
        with tf.variable_scope(scope):
            self.W = tf.get_variable(name = "Embeddings", shape = [instances, embedding_size], initializer = tf.glorot_uniform_initializer(seed = self.seed))
            tf.summary.scalar('mean', tf.reduce_mean(self.W))
    
    def __call__(self, ind):
        return tf.gather(self.W, ind)
