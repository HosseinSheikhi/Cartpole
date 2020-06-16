import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from ActorCriticNet import ActorCriticNet
from log_metric import A2CMetric
from Memory import Memory
import datetime

"""
 The implementation of the Advantage Actor Critic Algorithm for the Cartpole Environment
 For the fine tuning:
    Tune the number of hidden layers or num of units in each layer
    Tune the Lookahead value
    Tune the Adam optimizer learning_rate
"""

GAMMA = 0.99  # discount factor
LOOKAHEAD = 5  # Lookahead is equivalent to the t_max value in the A3C paper
EPSILON = tf.keras.backend.epsilon()

LOGGING = False
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
critic_loss_log_dir = 'logs/gradient_tape/' + current_time + '/critic_' + str(LOOKAHEAD)
actor_loss_log_dir = 'logs/gradient_tape/' + current_time + '/actor_' + str(LOOKAHEAD)
a2c_reward_log_dir = 'logs/gradient_tape/' + current_time + '/a2c_reward_' + str(LOOKAHEAD)


class A2CAgent:
    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states

        self.a2c_network = ActorCriticNet(self.num_actions, self.num_states).create_network()
        self.optimizer = keras.optimizers.Adam(0.001)
        self.critic_loss = keras.losses.Huber()

        if LOGGING:
            self.critic_loss_writer = tf.summary.create_file_writer(critic_loss_log_dir)
            self.actor_loss_writer = tf.summary.create_file_writer(actor_loss_log_dir)
            self.a2c_reward_writer = tf.summary.create_file_writer(a2c_reward_log_dir)
            self.critic_loss_metric = A2CMetric()
            self.actor_loss_metric = A2CMetric()
            self.a2c_reward_metric = A2CMetric()
            self.epoch = 0
            self.episode_reward = 1

    def act(self, state):
        action_prob, _ = self.a2c_network(state)
        return np.random.choice(self.num_actions, p=np.squeeze(action_prob))

    def prepare_train(self, memory, next_state):
        actions_one_hot = tf.one_hot(memory.action_history, depth=self.num_actions)
        batch_state = memory.state_history[0]
        for i in range(1, len(memory.state_history)):
            batch_state = tf.concat([batch_state, memory.state_history[i]], 0)

        self.train(batch_state, next_state, memory.reward_history, actions_one_hot)

    def train(self, states_in_batch, next_state, rewards, actions):
        tvs = self.a2c_network.trainable_variables
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]

        with tf.GradientTape(persistent=True) as tape:
            action_probs, critic_value = self.a2c_network(states_in_batch)
            roll_back_reward = 0
            targets = []
            terminal = True
            if len(next_state) > 0:  # if episode is not done
                next_state = tf.convert_to_tensor(next_state)
                next_state = tf.expand_dims(next_state, 0)
                _, critic = self.a2c_network(next_state)
                roll_back_reward = critic[0, 0]
                terminal = False

            for r in reversed(rewards):
                roll_back_reward = r + GAMMA * roll_back_reward
                targets.insert(0, roll_back_reward)

            if LOGGING:
                self.log(rewards, terminal)

            for i in range(len(targets)):
                critic_loss = self.critic_loss(tf.expand_dims(targets[i], 0), tf.expand_dims(critic_value[i, 0], 0))
                actor_loss = -tf.reduce_mean(actions[i] * tf.math.log((EPSILON + action_probs[i])))
                advantage = targets[i] - critic_value[i]
                loss = actor_loss * advantage + critic_loss

                if LOGGING:
                    self.critic_loss_metric.update_state(tf.squeeze(critic_loss))
                    self.actor_loss_metric.update_state(tf.squeeze(actor_loss * advantage))

                grads = tape.gradient(loss, self.a2c_network.trainable_variables)
                accum_grads = [accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]

        del tape
        self.optimizer.apply_gradients(zip(accum_grads, self.a2c_network.trainable_variables))

    def log(self, reward, terminal):
        self.episode_reward += sum(reward)
        if terminal:  # means episode ends
            self.epoch += 1
            self.a2c_reward_metric.update_state(self.episode_reward)  #
            with self.critic_loss_writer.as_default():
                tf.summary.scalar('critic_loss', self.critic_loss_metric.result(), step=self.epoch)
                tf.summary.scalar('actor_loss', self.actor_loss_metric.result(), step=self.epoch)
                tf.summary.scalar('a2c_reward', self.a2c_reward_metric.result(), step=self.epoch)
            self.critic_loss_metric.reset_states()
            self.actor_loss_metric.reset_states()
            self.a2c_reward_metric.reset_states()
            self.episode_reward = 1


def run():
    env = gym.make('CartPole-v0')
    episode_num = 1
    episode_done = True
    a2c_agent = A2CAgent(env.action_space.n, env.observation_space.shape[0])
    memory = Memory()
    episode_reward = 1

    while True:
        if episode_done:
            state = env.reset()
            next_state = []  # next state will pass to initialize the accumulated reward
            episode_done = False
            template = 'episode num {}  ends after {} time steps'
            print(template.format(episode_num, episode_reward))
            episode_num += 1
            episode_reward = 1

        for i in range(LOOKAHEAD):
            env.render()

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            action = a2c_agent.act(state)
            memory.store_state_action(state, action)
            state, reward, episode_done, _ = env.step(action)
            if episode_done and episode_reward < 198:
                reward = -1
            memory.store_reward(reward)
            episode_reward += reward

            next_state = state
            if episode_done:
                next_state = []  # if episode is done next state is None,
                break

        a2c_agent.prepare_train(memory, next_state)
        memory.clear()



def main():
    run()


if __name__ == '__main__':
    main()
