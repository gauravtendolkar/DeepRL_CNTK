import numpy as np
import random

class SimpleReplayBuffer:

    samples = []

    def __init__(self, capacity):
        self.capacity = capacity
        self.mem_counter = 0

    def add(self, sample):
        self.mem_counter += 1
        if self.mem_counter >= self.capacity:
            self.samples.pop(0)
            self.mem_counter -= 1
        self.samples.append(sample)

    def sample(self, size):
        n =  min(self.mem_counter, size)
        return random.sample(self.samples, n)

    def is_full(self):
        return self.mem_counter >= self.capacity-1


