from ActorCriticNet import ActorCriticNet
from WorkerAgent import WorkerAgent
import gym
import tensorflow as tf
import multiprocessing
from ray.experimental.queue import Queue
import ray

tf.config.set_visible_devices([], 'GPU')


class MasterAgent:
    def __init__(self):
        env = gym.make('CartPole-v0')
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.actor_critic_net = ActorCriticNet(self.num_actions, self.num_states).create_network()
        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        res_queue = Queue()
        workers = [WorkerAgent.remote(self.num_actions,
                                      self.num_states,
                                      i,
                                      res_queue) for i in range(multiprocessing.cpu_count())]
        workers_results = [worker.run.remote() for worker in workers]

        average_reward = []
        while True:
            #print(len(res_queue))
            #print(len(gradients_queue))
            if not res_queue.empty():
                average_reward.append(res_queue.get())
                if len(average_reward) == multiprocessing.cpu_count():
                    template = "Average reward: {} "
                    print(template.format(sum(average_reward) / len(average_reward)))
                    average_reward.clear()

    def test(self):
        pass


if __name__ == '__main__':
    a3c_agent = MasterAgent()
    a3c_agent.train()
