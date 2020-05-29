from denseLayer import DenseLayer
import tensorflow as tf


class CriticNetwork(tf.keras.Model):
    def __init__(self, shared_blk):
        super(CriticNetwork, self).__init__()
        self.shared_blk = shared_blk
        self.dense_1 = DenseLayer(16)
        self.critic = DenseLayer(1)

    def call(self, inputs):
        shared_out = self.shared_blk(inputs)
        dense_out = self.dense_1(shared_out)
        critic = self.critic(tf.nn.relu(dense_out))
        return critic
