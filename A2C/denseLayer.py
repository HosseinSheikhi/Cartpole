import tensorflow as tf
from tensorflow.keras.layers import Layer


class DenseLayer(Layer):
    def __init__(self, num_units):
        super(DenseLayer, self).__init__()
        self.num_units = num_units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.num_units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.num_units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
