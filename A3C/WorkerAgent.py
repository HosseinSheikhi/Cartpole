import numpy as np
import tensorflow as tf
import threading
from ActorCriticNet import ActorCriticNet
from Trajectory import Trajectory
import gym
import datetime
from log_metric import A3CMetric

LOOKAHEAD = 5
GAMMA = 0.99
EPSILON = tf.keras.backend.epsilon()

LOGGING = False
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class WorkerAgent(threading.Thread):
    def __init__(self, num_actions, num_states, worker_id, global_actor_critic, global_optimizer, res_queue):
        super(WorkerAgent, self).__init__()

        self.num_actions = num_actions
        self.num_states = num_states
        self.local_actor_critic = ActorCriticNet(self.num_actions, self.num_states).create_network()
        self.global_actor_critic = global_actor_critic
        self.global_optimizer = global_optimizer
        self.res_queue = res_queue
        self.worker_id = worker_id
        self.critic_loss = tf.keras.losses.Huber()

        # Update local model with new weights
        self.local_actor_critic.set_weights(self.global_actor_critic.get_weights())

        if LOGGING:
            critic_loss_log_dir = 'logs/gradient_tape/' + current_time + '/critic_' + str(self.worker_id)
            actor_loss_log_dir = 'logs/gradient_tape/' + current_time + '/actor_' + str(self.worker_id)
            a3c_reward_log_dir = 'logs/gradient_tape/' + current_time + '/a3c_reward_' + str(self.worker_id)
            self.critic_loss_writer = tf.summary.create_file_writer(critic_loss_log_dir)
            self.actor_loss_writer = tf.summary.create_file_writer(actor_loss_log_dir)
            self.a3c_reward_writer = tf.summary.create_file_writer(a3c_reward_log_dir)
            self.critic_loss_metric = A3CMetric()
            self.actor_loss_metric = A3CMetric()
            self.a3c_reward_metric = A3CMetric()
            self.epoch = 0
            self.episode_reward = 1

    def act(self, state):
        policy, _ = self.local_actor_critic(state)
        return np.random.choice(self.num_actions, p=np.squeeze(policy))

    def train(self, trajectory, next_state):

        states_batch = trajectory.state_history[0]
        for i in range(1, len(trajectory.state_history)):
            states_batch = tf.concat([states_batch, trajectory.state_history[i]], 0)
        actions_one_hot = tf.one_hot(trajectory.action_history, depth=self.num_actions)

        tvs = self.local_actor_critic.trainable_variables
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]

        with tf.GradientTape(persistent=True) as tape:
            roll_back_reward = 0
            targets = []
            terminal = True
            if len(next_state) > 0:
                next_state = tf.convert_to_tensor(next_state)
                next_state = tf.expand_dims(next_state, 0)
                _, critic = self.local_actor_critic(next_state)
                roll_back_reward = critic[0, 0]
                terminal = False

            for r in reversed(trajectory.reward_history):
                roll_back_reward = r + GAMMA * roll_back_reward
                targets.insert(0, roll_back_reward)

            if LOGGING:
                self.log(trajectory.reward_history, terminal)

            action_probs, critic = self.local_actor_critic(states_batch)
            for i in range(len(targets)):
                critic_loss = self.critic_loss(tf.expand_dims(targets[i], 0), tf.expand_dims(critic[i, 0], 0))
                actor_loss = -tf.reduce_mean(actions_one_hot[i] * tf.math.log((EPSILON + action_probs[i])))
                advantage = targets[i] - critic[i]
                total_loss = actor_loss * advantage + critic_loss

                if LOGGING:
                    self.critic_loss_metric.update_state(tf.squeeze(critic_loss))
                    self.actor_loss_metric.update_state(tf.squeeze(actor_loss * advantage))

                grads = tape.gradient(total_loss, self.local_actor_critic.trainable_variables)
                accum_grads = [accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]
        del tape
        self.global_optimizer.apply_gradients(zip(accum_grads, self.global_actor_critic.trainable_variables))
        # Update local model with new weights
        self.local_actor_critic.set_weights(self.global_actor_critic.get_weights())

    def run(self):
        env = gym.make("CartPole-v0")
        episode_done = True
        trajectory = Trajectory()
        next_state = []
        episode_num = 0
        episode_reward = 1

        while True:
            if episode_done:
                state = env.reset()
                episode_done = False
                template = "in worker {}, episode {} done after {} steps"
                print(template.format(self.worker_id, episode_num, episode_reward))
                self.res_queue.put(episode_reward)
                episode_num += 1
                episode_reward = 1

            for i in range(LOOKAHEAD):
                # env.render()
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                action = self.act(state)
                trajectory.store(s=state, a=action)
                state, reward, episode_done, _ = env.step(action)
                if episode_done:
                    reward = -1

                trajectory.store(r=reward)
                episode_reward += reward
                next_state = state

                if episode_done:
                    next_state = []
                    break

            self.train(trajectory, next_state)
            trajectory.clear()

    def log(self, reward, terminal):
        self.episode_reward += sum(reward)
        if terminal:  # means episode ends
            self.epoch += 1
            self.a3c_reward_metric.update_state(self.episode_reward)  #
            with self.critic_loss_writer.as_default():
                tf.summary.scalar('critic_loss', self.critic_loss_metric.result(), step=self.epoch)
                tf.summary.scalar('actor_loss', self.actor_loss_metric.result(), step=self.epoch)
                tf.summary.scalar('a2c_reward', self.a3c_reward_metric.result(), step=self.epoch)
            self.critic_loss_metric.reset_states()
            self.actor_loss_metric.reset_states()
            self.a3c_reward_metric.reset_states()
            self.episode_reward = 1
