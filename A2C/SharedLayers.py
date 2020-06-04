from denseLayer import DenseLayer
import tensorflow as tf


class SharedLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SharedLayer, self).__init__()
        self.dense_1 = DenseLayer(128)
        self.dense_2 = DenseLayer(128)

    def call(self, inputs):
        x_1 = self.dense_1(inputs)
        x_2 = self.dense_2(tf.nn.relu(x_1))
        return tf.nn.relu(x_2)
