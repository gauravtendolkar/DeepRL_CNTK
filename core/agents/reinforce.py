import random
import numpy as np
from policies.nn_policies import REINFORCENNPolicy
from utils.buffers import EpisodicBuffer
import warnings

warnings.filterwarnings("ignore")

MAX_EPSILON = 1.0
REPLAY_BUFFER_CAPACITY = 10000
STEPS_BEFORE_EPSILON_DECAY = 500000
BATCH_SIZE = 32
TERMINAL_STATE = None
DISCOUNT_FACTOR = 0.99


class RAMAgent:
    steps = 0

    def __init__(self, num_actions, observation_space_shape, pretrained_policy=None,
                 *args, **kwargs):
        self.actor_policy = REINFORCENNPolicy(name='Evaluation Network',
                                              observation_space_shape=observation_space_shape,
                                              num_actions=num_actions, pretrained_policy=pretrained_policy)
        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        self.memory = EpisodicBuffer(discount_factor=DISCOUNT_FACTOR)

    def act(self, current_state):
        """
        Decide action to take based on current state and exploration strategy. Initailly we explore (i.e. choose uniformly random action) 100% of the times
        But epsilon decays as we gain experience and we start taking policy determined actions, i.e. the action with highest Q value at current state, as determined by evaluation network
        """
        return np.choice(range(self.num_actions), p=self.actor_policy.predict(current_state))

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

    def learn(self):
        episode = self.memory.get_episode()
        num_steps = len(episode)
        cumulative_future_rewards = []

        rewards = [e[2] for e in episode]
        actions = [e[1] for e in episode]
        states = [e[0] for e in episode]

        for t in range(num_steps):
            cumulative_future_reward = 0
            for step, reward in enumerate(rewards[t:]):
                cumulative_future_reward += (DISCOUNT_FACTOR**step) * reward
            cumulative_future_rewards.append(cumulative_future_reward)

        std_cumulative_future_rewards = np.std(cumulative_future_rewards)
        if std_cumulative_future_rewards > 0:
            cumulative_future_rewards = (cumulative_future_rewards - np.mean(cumulative_future_rewards))/std_cumulative_future_rewards

        self.actor_policy.optimise(states, actions, cumulative_future_rewards)