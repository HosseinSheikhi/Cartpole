class Memory:
    def __init__(self):
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def store(self, s=None, a=None, r=None):
        if s is not None:
            self.state_history.append(s)
        if a is not None:
            self.action_history.append(a)
        if r is not None:
            self.reward_history.append(r)

    def clear(self):
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()