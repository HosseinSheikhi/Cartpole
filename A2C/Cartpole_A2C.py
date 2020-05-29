import gym
from ActorCritic import A2CAgent
import numpy as np

file = open("C:/Users/Hossein/Desktop/a2c-v0.txt", "a")
MAX_TRAINING_STP = 20000


def cartpole():
    env = gym.make("CartPole-v0")
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    a2c_agent = A2CAgent(action_space_size, None)
    episode_num = 0
    action_num = 0
    finish = 0
    while True:
        episode_num += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space_size])
        t = 0
        while True:
            finish += 1
            env.render()
            action = a2c_agent.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space_size])

            a2c_agent.train(state, action, state_next, reward, terminal)
            state = state_next
            t += 1
            action_num += 1
            if terminal:
                print("Episode {} finished after {} timesteps, steps remaining {}".format(episode_num, t, MAX_TRAINING_STP-finish))
                file.write(str(episode_num) + "  " + str(t) + "\n")
                break
        if finish > MAX_TRAINING_STP:
            file.close()
            break


def main():
    cartpole()


if __name__ == "__main__":
    main()
