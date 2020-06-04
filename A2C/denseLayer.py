import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons):
        super(DenseLayer, self).__init__()
        self.num_nuerons = num_neurons

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.num_nuerons), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.num_nuerons,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
