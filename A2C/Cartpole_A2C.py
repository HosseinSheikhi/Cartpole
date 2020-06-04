import gym
from ActorCritic import A2CAgent
import numpy as np
from collections import deque
MAX_TRAINING_EPISODE = 2000
LOOK_AHEAD = 3


def cartpole():
    env = gym.make("CartPole-v0")
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    a2c_agent = A2CAgent(action_space_size)
    episode_num = 0
    n_step_stack = deque(maxlen=LOOK_AHEAD)
    while True:
        episode_num += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space_size])
        episode_steps = 0
        while True:

            #env.render()
            action = a2c_agent.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space_size])
            n_step_stack.append([state, action, state_next, reward, terminal])
            a2c_agent.train(n_step_stack)
            state = state_next

            episode_steps += 1
            if terminal:
                print("Episode {} finished after {} time steps, steps remaining {}".format(episode_num, episode_steps, MAX_TRAINING_EPISODE - episode_num))
                break
        if episode_num > MAX_TRAINING_EPISODE:
            break


def main():
    cartpole()


if __name__ == "__main__":
    main()
