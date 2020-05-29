from denseLayer import DenseLayer
import tensorflow as tf


class ActorNetwork(tf.keras.Model):
    def __init__(self, num_actions, shared_blk):
        super(ActorNetwork, self).__init__()
        self.num_actions = num_actions
        self.shared_blk = shared_blk
        self.dense_1 = DenseLayer(32)
        self.actor = DenseLayer(self.num_actions)

    def call(self, inputs, training=None, mask=None):
        shared_out = self.shared_blk(inputs)
        dense_out = self.dense_1(shared_out)
        actions = self.actor(tf.nn.relu(dense_out))
        return tf.nn.softmax(actions)
