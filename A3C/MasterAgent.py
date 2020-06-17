from ActorCriticNet import ActorCriticNet
from WorkerAgent import WorkerAgent
import gym
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue


class MasterAgent:
    def __init__(self):
        env = gym.make('CartPole-v0')
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.actor_critic_net = ActorCriticNet(self.num_actions, self.num_states).create_network()
        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        res_queue = Queue()
        workers = [WorkerAgent(self.num_actions,
                               self.num_states,
                               i,
                               self.actor_critic_net,
                               self.optimizer,
                               res_queue) for i in range(multiprocessing.cpu_count())]
        for worker in workers:
            worker.start()

        average_reward = []
        while True:
            if not res_queue.empty():
                average_reward.append(res_queue.get())
                if len(average_reward) == multiprocessing.cpu_count():
                    template = "Average reward: {} "
                    print(template.format(sum(average_reward) / len(average_reward)))
                    average_reward.clear()

        [w.join() for w in workers]

    def test(self):
        pass


if __name__ == '__main__':
    a3c_agent = MasterAgent()
    a3c_agent.train()
