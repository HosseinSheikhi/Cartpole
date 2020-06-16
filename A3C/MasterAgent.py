from ActorCriticNet import ActorCriticNet
from WorkerAgent import WorkerAgent
import gym
import tensorflow as tf


class MasterAgent:
    def __init__(self):
        env = gym.make('CartPole-v0')
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.actor_critic_net = ActorCriticNet(self.num_actions, self.num_states).create_network()
        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        worker = WorkerAgent(self.num_actions, self.num_states, self.actor_critic_net, self.optimizer, 1)
        worker.start()

        worker2 = WorkerAgent(self.num_actions, self.num_states, self.actor_critic_net, self.optimizer, 2)
        worker2.start()

    def test(self):
        pass


if __name__ == '__main__':
    a3c_agent = MasterAgent()
    a3c_agent.train()
