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

        self.actor_optimizer = tf.optimizers.Adam(learning_rate=0.00025)  # 0.0005
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=0.0005)  # 0.001

    def act(self, state):
        action_prob = self.actor_network.predict(state)
        action = np.random.choice(self.num_actions, p=action_prob[0])
        return action

    def calculate_target(self, states, next_states, rewards, terminals):
        LOOK_AHEAD = len(states)

        look_ahead_reward = 0
        terminal_happend = 0
        for k in range(LOOK_AHEAD):
            look_ahead_reward += np.power(GAMMA, k) * rewards[k]
            if terminals[k]:
                terminal_happend = 1
                break

        v_current_state = self.critic_network(states[0])
        v_next_state = self.critic_network(next_states[LOOK_AHEAD - 1])
        value_target = look_ahead_reward + np.power(GAMMA, k + 1) * v_next_state * (1 - terminal_happend)

        advantage = value_target - v_current_state
        return value_target, advantage

    def value_network_loss_function(self, y_true, y_pred):
        return tf.square(y_true - y_pred)

    def policy_network_loss_function(self, y_true, y_pred, advantage):
        y_pred = tf.clip_by_value(y_pred, clip_value_min=EPSILON, clip_value_max=1 - EPSILON)
        entropy = tf.reduce_sum(y_pred * tf.math.log(y_pred))
        policy_loss = tf.reduce_mean(tf.one_hot(y_true, self.num_actions) * tf.math.log(y_pred))
        loss = -policy_loss * tf.stop_gradient(advantage) - 0.01 * entropy
        return loss

    @tf.function
    def critic_network_optimization(self, x, y):
        with tf.GradientTape() as g:
            prediction = self.critic_network(x)
            loss = self.value_network_loss_function(y, prediction)

        trainable_variables = self.critic_network.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, trainable_variables))

    @tf.function
    def actor_network_optimization(self, x, y, advantage):
        with tf.GradientTape() as g:
            prediction = self.actor_network(x)
            loss = self.policy_network_loss_function(y, prediction, advantage)

        trainable_variables = self.actor_network.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, trainable_variables))

    def train(self, n_step_stack):
        current_states = []
        next_states = []
        rewards = []
        actions = []
        terminals = []
        for _, (s, a, s_, r, d) in enumerate(n_step_stack):
            current_states.append(s)
            actions.append(a)
            next_states.append(s_)
            rewards.append(r)
            terminals.append(d)

        target, advantage = self.calculate_target(current_states, next_states, rewards, terminals)

        value_train_ds = tf.data.Dataset.from_tensor_slices((current_states[0], target)).batch(BATCH_SIZE)

        policy_train_ds = tf.data.Dataset.from_tensor_slices((current_states[0], tf.Variable([actions[0]], dtype=tf.int32))).batch(
            BATCH_SIZE)

        for (x, y) in value_train_ds:
            self.critic_network_optimization(x, y)

        for (x, y) in policy_train_ds:
            self.actor_network_optimization(x, y, tf.Variable(advantage))
