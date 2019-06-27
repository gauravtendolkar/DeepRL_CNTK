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


class FrameStacker:

    def __init__(self, stack_size, frame_shape):
        self.stack_size = stack_size
        self.frame_shape = frame_shape
        self.buffer = np.zeros(shape=(stack_size, *(frame_shape)))
        self.full = False

    def add_frame(self, frame):
        if not self.full:
            for i in range(self.stack_size):
                self.buffer[i, :] = frame
            self.full = True
        else:
            self.buffer[:-1,:] = self.buffer[1:,:]
            self.buffer[-1,:] = frame
        return np.copy(self.buffer)

    def reset(self):
        self.buffer = np.zeros(shape=(self.stack_size, *(self.frame_shape)))
        self.full = False


class EpisodicBuffer:

    episode = []

    def __init__(self):
        pass

    def add(self, sample):
        self.episode.append(sample)

    def get_episode(self):
        return self.episode

    def reset(self):
        self.episode = []



