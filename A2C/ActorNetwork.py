from denseLayer import DenseLayer
import tensorflow as tf


class ActorNetwork(tf.keras.Model):
    def __init__(self, num_actions, shared_blk):
        super(ActorNetwork, self).__init__()
        self.num_actions = num_actions
        self.shared_blk = shared_blk
        self.actor_layer = DenseLayer(self.num_actions)

    def call(self, inputs, training=None, mask=None):
        shared_out = self.shared_blk(inputs)
        actions = self.actor_layer(shared_out)
        return tf.nn.softmax(actions)
