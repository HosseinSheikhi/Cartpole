class Memory:
    def __init__(self):
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def store_state_action(self, s, a):
        self.state_history.append(s)
        self.action_history.append(a)

    def store_reward(self, r):
        self.reward_history.append(r)

    def clear(self):
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()