import numpy as np
import tensorflow as tf
import threading
from ActorCriticNet import ActorCriticNet
from Trajectory import Trajectory
import gym

LOOKAHEAD = 5
GAMMA = 0.99
EPSILON = tf.keras.backend.epsilon()


class WorkerAgent(threading.Thread):
    def __init__(self, num_actions, num_states, global_actor_critic, global_optimizer, worker_id):
        super(WorkerAgent, self).__init__()

        self.num_actions = num_actions
        self.num_states = num_states
        self.local_actor_critic = ActorCriticNet(num_actions, num_states).create_network()
        self.global_actor_critic = global_actor_critic
        self.global_optimizer = global_optimizer
        self.worker_id = worker_id
        self.critic_loss = tf.keras.losses.Huber()

        # Update local model with new weights
        self.local_actor_critic.set_weights(self.global_actor_critic.get_weights())

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
            if len(next_state) > 0:
                next_state = tf.convert_to_tensor(next_state)
                next_state = tf.expand_dims(next_state, 0)
                _, critic = self.local_actor_critic(next_state)
                roll_back_reward = critic[0, 0]

            for r in reversed(trajectory.reward_history):
                roll_back_reward = r + GAMMA * roll_back_reward
                targets.insert(0, roll_back_reward)

            action_probs, critic = self.local_actor_critic(states_batch)
            for i in range(len(targets)):
                critic_loss = self.critic_loss(tf.expand_dims(targets[i], 0), tf.expand_dims(critic[i, 0], 0))
                actor_loss = -tf.reduce_mean(actions_one_hot[i] * tf.math.log((EPSILON + action_probs[i])))
                advantage = targets[i] - critic[i]
                total_loss = actor_loss * advantage + critic_loss
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
                episode_num += 1
                episode_reward = 1

            for i in range(LOOKAHEAD):
                #env.render()
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                action = self.act(state)
                trajectory.store_state_action(state, action)

                state, reward, episode_done, _ = env.step(action)
                if episode_done:
                    reward = -1

                trajectory.store_reward(reward)
                episode_reward += reward
                next_state = state

                if episode_done:
                    next_state = []
                    break

            self.train(trajectory, next_state)
            trajectory.clear()
