import gym
from ActorCritic import A2CAgent
import numpy as np
from collections import deque
file = open("/home/hossein/Desktop/cartpole-a2c-5step_v2.txt", "a")
MAX_TRAINING_EPISODE = 600
LOOK_AHEAD = 3


def cartpole():
    env = gym.make("CartPole-v0")
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    a2c_agent = A2CAgent(action_space_size, None)
    episode_num = 0
    action_num = 0
    finish = 0
    n_step_stack = deque(maxlen=LOOK_AHEAD)
    while True:
        episode_num += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space_size])
        t = 0
        while True:
            action_num += 1
            finish += 1
            env.render()
            action = a2c_agent.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space_size])
            n_step_stack.append([state, action, state_next, reward, terminal])
            if action_num > LOOK_AHEAD:
                a2c_agent.train(n_step_stack)

            state = state_next
            t += 1
            if terminal:
                print("Episode {} finished after {} timesteps, steps remaining {}".format(episode_num, t,
                                                                                          MAX_TRAINING_EPISODE - episode_num))
                file.write(str(episode_num) + "  " + str(t) + "\n")
                break
        if episode_num > MAX_TRAINING_EPISODE:
            file.close()
            break


def main():
    cartpole()


if __name__ == "__main__":
    main()
