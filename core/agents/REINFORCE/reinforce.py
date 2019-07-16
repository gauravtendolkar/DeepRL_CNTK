import numpy as np
from policies.cnn_policies_2 import ActorCNNPolicy, CriticCNNPolicy
from utils.buffers import SimpleBuffer, FrameSubtractor
from agents.a2c.hyperparams import BATCH_SIZE, DISCOUNT_FACTOR


class REINFORCEAgent:
    steps = 0

    def __init__(self, num_actions, observation_space_shape, actor_pretrained_policy=None, *args, **kwargs):
        self.actor_policy = ActorCNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                           num_actions=num_actions, pretrained_policy=actor_pretrained_policy)

        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        self.memory = SimpleBuffer()
        self.frame_preprocessor = FrameSubtractor()

    def act(self, current_state):
        probabilities = np.squeeze(self.actor_policy.predict(current_state))
        action = np.random.choice(range(self.num_actions), 1, p=probabilities)
        return action

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

    def learn(self):
        episode = self.memory.get_data()
        n = len(episode)

        current_states = np.array([e[0] for e in episode])
        actions = np.array([[e[1]] for e in episode])
        rewards = [e[2] for e in episode]

        # Calculate returns from rewards
        returns = [0]*n
        for i in range(n):
            ret = 0
            for t in range(i, n):
                ret += (DISCOUNT_FACTOR ** (t - i)) * rewards[t]
            returns[i] = [ret]

        returns = np.array(np.divide((returns - np.mean(returns)), np.std(returns) + 0.000001))

        for i in range((n // BATCH_SIZE) + 1):
            start, end = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, n)
            if start < end:
                self.actor_policy.optimise(current_states[start:end], returns[start:end], actions[start:end])

        self.memory.reset()