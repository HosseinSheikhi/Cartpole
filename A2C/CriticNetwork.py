from denseLayer import DenseLayer
import tensorflow as tf


class CriticNetwork(tf.keras.Model):
    def __init__(self, shared_blk):
        super(CriticNetwork, self).__init__()
        self.shared_blk = shared_blk
        self.critic_layer = DenseLayer(1)

    def call(self, inputs):
        shared_out = self.shared_blk(inputs)
        critic = self.critic_layer(shared_out)
        return critic
