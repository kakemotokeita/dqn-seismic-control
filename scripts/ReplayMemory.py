import random
from Transition import Transition

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Transitionを記録する
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # ランダムなサンプルをリターン
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)