#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:25:18 2020

@author: hossein
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import gym
from collections import deque
import numpy as np
import random as rnd
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
UPDATE_AFTER = 1000
DISCOUNT=0.95
MAX_ACTION=35000



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  
        
        """
        in each step in training phase, the agent will do an action, to do the action,
        agent needs the Q_VALUES which is the QNetwork (obviousely it's maximum)
        so for each action. the QNetwork.predict() will be called
        On the other hand after each action the QNetwork will tries to fit it self just
        by one input! There are two problem on this:
                1- Fitting a NN just by one input is not a good idea. We know it is better
                   to fitting on a BATCH of data (A sequence of data)
                2- If we do predict-fit-predict-fit we do not have consistency on predicts
        So it is better to use two QNetwork:
                1-model    2-target_model
        and also using experience replay
        The agent use for target_model for predicts and the 
        model we be trained on BATCH thanks to the experience reply
        eperience replay is also for planning (like DYNAQ on RL specialization)
        we will copy model to target_model after UPDATE_AFTER actions
        """
        self.model = self.create_QNetwork()
        self.target_model = self.create_QNetwork()
        self.target_model.set_weights(self.model.get_weights())          
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
  
        self.tau=1
        self.target_update_after_counter = 0
        
    def create_QNetwork(self):
        model = Sequential()
        model.add(Dense(16,input_dim=self.state_size))
        model.add(Activation('relu'))
        
        model.add(Dense(16))
        model.add(Activation('relu'))
       
        model.add(Dense(16))
        model.add(Activation('relu'))
        
        model.add(Dense(16))
        model.add(Activation('relu'))
        
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        
        model.compile(optimizer=Adam(lr=0.001,decay=0.00001),loss="mse",metrics=['accuracy'])
        
        return model

    def get_Qvalues(self,state):
        return self.model.predict(state)

    def softmax(self,Qvalues):
        preferences = Qvalues/self.tau
        max_preference = np.amax(Qvalues, axis=1)/self.tau
        reshaped_max_preference = max_preference.reshape((-1, 1))
        
        # Compute the numerator, i.e., the exponential of the preference - the max preference.
        exp_preferences = np.exp(preferences - reshaped_max_preference)
        # Compute the denominator, i.e., the sum over the numerator along the actions axis.
        sum_of_exp_preferences = np.sum(exp_preferences,axis=1)
        
        reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
        action_probs = exp_preferences / reshaped_sum_of_exp_preferences
        action_probs = action_probs.squeeze()
        return action_probs

        
    def act(self,state):
        Qvalues = self.get_Qvalues(state)
        actions_probability = self.softmax(Qvalues)
        action = np.random.choice(self.action_size, p=actions_probability.squeeze())
        return action


    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
            
        
    def train(self,terminal):
        if len(self.replay_memory)<MIN_REPLAY_MEMORY_SIZE:
            return
        for i in range(2):
            minibatch = rnd.sample(self.replay_memory,MINIBATCH_SIZE)
            current_states = np.array([transition[0] for transition in minibatch])
            current_Qvalues_list = self.model.predict(current_states.squeeze())
            
            next_states = np.array([transition[3] for transition in minibatch])
            next_Qvalues_list = self.target_model.predict(next_states.squeeze())
            next_actions_prob = self.softmax(next_Qvalues_list)
            X_train=[]
            Y_train=[]
            
            for index, (current_state , action , reward , next_state , done) in enumerate(minibatch):
                if not done:
                    future_reward = np.inner(next_Qvalues_list[index],next_actions_prob[index])
                    desired_q = reward + DISCOUNT * future_reward
                else:
                    desired_q = reward
                    
                current_QValues = current_Qvalues_list[index]
                current_QValues[action]=desired_q
                
                X_train.append(current_state)
                Y_train.append(current_QValues)
                
            X_train=np.array(X_train)
            Y_train = np.array(Y_train)
            X_train = np.reshape(X_train,[len(minibatch),self.state_size])
            Y_train = np.reshape(Y_train,[len(minibatch),self.action_size])
            
            self.model.fit(X_train,Y_train,batch_size=MINIBATCH_SIZE,verbose=0)
            self.target_update_after_counter+=1
        
        if self.target_update_after_counter>UPDATE_AFTER and terminal:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_after_counter=0
            print("*Target model updated*")


file = open("/home/hossein/Desktop/expectedSARSA-v0.txt","a")            
def cartpole():
    env = gym.make("CartPole-v0")
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    dqn_agent = DQNAgent(observation_space_size, action_space_size)
    episode_num=0
    action_num=0
    while True:
        episode_num+=1
        state = env.reset()
        state = np.reshape(state, [1, observation_space_size])
        t=0
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
            t+=1
            action_num+=1
            if action_num>MAX_ACTION:
                file.close()
                env.close()
            if terminal:
                file.write(str(t)+"\n")
                print("Episode {} finished after {} timesteps".format(episode_num,t))
                break
            

def main():
    cartpole()


if __name__ =="__main__":
    main()
