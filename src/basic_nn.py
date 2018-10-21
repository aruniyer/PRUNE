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
        self.instances = instances
        self.scope = scope
        with tf.variable_scope(scope):
            self.W = tf.get_variable("Embeddings", [instances, embedding_size])
            tf.summary.scalar('mean', tf.reduce_mean(self.W))
    
    def __call__(self, ind):
        return tf.gather(self.W, ind)
