#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:35:16 2020

    in each step in training phase, the agent will do an action, to do the action,
    agent needs the Q_VALUES which is the QNetwork output (obviousely it's maximum).

    So for each action the QNetwork.predict() will be called

    On the other hand after each action the QNetwork will tries to fit itself just
    by one input! There are two problem on this:
            1- Fitting a NN just by one input is not a good idea. We know it is better
               to fitting on a BATCH of data (A sequence of data)
            2- If we do predict-fit-predict-fit we do not have consistency on predicts
    So it is better to use experience replay and also two QNetwork:
            1-model    2-target_model

    The agent will use the target_model for predictions and the
    model will be trained on BATCH thanks to the experience reply

    we will copy model to target_model after UPDATE_AFTER actions

@author: hossein
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from log_metric import DQNMetric
import gym
from collections import deque
import random as rnd
import numpy as np
import math
import datetime

tf.config.set_visible_devices([], 'GPU')

REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 16
UPDATE_TARGET_AFTER = 1000
DISCOUNT = 0.95
EPSILON_MIN = 0.05
EPSILON_DECAY = 200
MAX_ACTION = 35000

LOGGING = True
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dqn_reward_log_dir = 'logs/gradient_tape/' + current_time + '/dqn_reward_2'


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.epsilon = 1
        self.target_update_after_counter = 0
        self.step_counter = 0

        if LOGGING:
            self.dqn_reward_writer = tf.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()



    def create_qnetwork(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))

        model.compile(optimizer=Adam(lr=0.001, decay=0.00001), loss="mse", metrics=['accuracy'])

        return model

    def get_qvalues(self, state):
        return self.model.predict(state)

    def act(self, state):
        self.step_counter += 1
        self.epsilon = EPSILON_MIN + (1 - EPSILON_MIN) * math.exp(-1.0 * self.step_counter / EPSILON_DECAY)
        lucky = rnd.random()
        if lucky > (1 - self.epsilon):
            return rnd.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.get_qvalues(state))

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        data_in_mini_batch = rnd.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states)

        next_states = np.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_qvalues_list = self.target_model.predict(next_states)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, next_state, done) in enumerate(data_in_mini_batch):
            if not done:
                future_reward = np.max(next_qvalues_list[index])
                desired_q = reward + DISCOUNT * future_reward
            else:
                desired_q = reward

            current_q_values = current_qvalues_list[index]
            current_q_values[action] = desired_q

            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = np.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(tf.convert_to_tensor(x_train, tf.float32), tf.convert_to_tensor(y_train, tf.float32),
                       batch_size=MINIBATCH_SIZE, verbose=0)
        self.target_update_after_counter += 1

        if self.target_update_after_counter > UPDATE_TARGET_AFTER and terminal:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_after_counter = 0
            print("*Target model updated*")


def cartpole():
    env = gym.make("CartPole-v0")
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    dqn_agent = DQNAgent(observation_space_size, action_space_size)
    task_done = deque(maxlen=20)
    episode_num = 0
    action_num = 0
    while True:
        episode_num += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space_size])
        t = 0
        while True:
            env.render()
            action = dqn_agent.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space_size])
            transition = (state, action, reward, state_next, terminal)
            dqn_agent.update_replay_memory(transition)
            dqn_agent.train(terminal)
            state = state_next
            t += 1
            action_num += 1
            if sum(task_done) / (len(task_done)+1) > 195:
                env.close()
            if terminal:
                print("Episode {} finished after {} time steps".format(episode_num, t))
                task_done.append(t)
                if LOGGING:
                    dqn_agent.dqn_reward_metric.update_state(t)
                    with dqn_agent.dqn_reward_writer.as_default():
                        tf.summary.scalar('dqn_reward', dqn_agent.dqn_reward_metric.result(), step=episode_num)
                    dqn_agent.dqn_reward_metric.reset_states()

                break


def main():
    cartpole()


if __name__ == "__main__":
    main()
