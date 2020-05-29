from CriticNetwork import CriticNetwork
from ActorNetwork import ActorNetwork
from SharedLayers import SharedLayer
import numpy as np
import tensorflow as tf

BATCH_SIZE = 1
GAMMA = 0.99
EPSILON = tf.keras.backend.epsilon()


class A2CAgent:
    def __init__(self, num_actions, weight_address=None):

        self.num_actions = num_actions
        self.weight_address = weight_address

        shared_blk = SharedLayer()

        self.actor_network = ActorNetwork(self.num_actions, shared_blk)
        self.critic_network = CriticNetwork(shared_blk)

        self.policy_opt = tf.optimizers.Adam(learning_rate=0.001)  # 0.0005
        self.value_opt = tf.optimizers.Adam(learning_rate=0.005)  # 0.001

    def act(self, state):
        action_prob = self.actor_network.predict(state)
        action = np.random.choice(self.num_actions, p=action_prob[0])
        return action

    def calculate_target(self, state, next_state, reward, terminal):
        v_current_state = self.critic_network(state)
        v_next_state = self.critic_network(next_state)
        value_target = reward + GAMMA * v_next_state * (1 - int(terminal))
        advantage = value_target - v_current_state
        return value_target, advantage

    def value_network_loss_function(self, y_true, y_pred):
        return tf.square(y_true - y_pred)

    def policy_network_loss_function(self, y_true, y_pred, advantage):
        action_prob = tf.nn.softmax(y_pred)
        action_prob = tf.clip_by_value(action_prob, clip_value_min=EPSILON, clip_value_max=1 - EPSILON)
        entropy = tf.reduce_sum(action_prob * tf.math.log(action_prob))
        policy_loss = tf.reduce_mean(tf.one_hot(y_true, self.num_actions) * tf.math.log(action_prob))
        loss = -policy_loss * tf.stop_gradient(advantage) - 0.01 * entropy
        # tf.print(y_true, advantage, loss)
        return loss

    @tf.function
    def value_network_optimization(self, x, y):
        with tf.GradientTape() as g:
            prediction = self.critic_network(x)
            loss = self.value_network_loss_function(y, prediction)

        trainable_variables = self.critic_network.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        self.value_opt.apply_gradients(zip(gradients, trainable_variables))

    @tf.function
    def policy_network_optimization(self, x, y, advantage):
        with tf.GradientTape() as g:
            prediction = self.actor_network(x)
            loss = self.policy_network_loss_function(y, prediction, advantage)

        trainable_variables = self.actor_network.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        self.policy_opt.apply_gradients(zip(gradients, trainable_variables))

    def train(self, state, action, next_state, reward, terminal):

        target, advantage = self.calculate_target(state, next_state, reward, terminal)

        value_train_ds = tf.data.Dataset.from_tensor_slices((state, target)).batch(BATCH_SIZE)

        policy_train_ds = tf.data.Dataset.from_tensor_slices((state, tf.Variable([action], dtype=tf.int32))).batch(
            BATCH_SIZE)

        for (x, y) in value_train_ds:
            self.value_network_optimization(x, y)

        for (x, y) in policy_train_ds:
            self.policy_network_optimization(x, y, tf.Variable(advantage))
