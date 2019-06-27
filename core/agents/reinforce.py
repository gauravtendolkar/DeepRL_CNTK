import random
import numpy as np
from policies.cnn_policies import StackedFrameCNNPolicy
from utils.buffers import EpisodicBuffer
import warnings

warnings.filterwarnings("ignore")

MAX_EPSILON = 1.0
REPLAY_BUFFER_CAPACITY = 10000
STEPS_BEFORE_EPSILON_DECAY = 500000
BATCH_SIZE = 32
TERMINAL_STATE = None
DISCOUNT_FACTOR = 0.99


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, num_actions, observation_space_shape, pretrained_policy=None, explore=True,
                 *args, **kwargs):
        self.actor_policy = StackedFrameCNNPolicy(name='Evaluation Network', num_frames_to_stack=4,
                                                       observation_space_shape=observation_space_shape,
                                                       num_actions=num_actions, pretrained_policy=pretrained_policy)
        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        self.memory = EpisodicBuffer()
        if not explore:
            self.epsilon = 0.0

    def act(self, current_state):
        """
        Decide action to take based on current state and exploration strategy. Initailly we explore (i.e. choose uniformly random action) 100% of the times
        But epsilon decays as we gain experience and we start taking policy determined actions, i.e. the action with highest Q value at current state, as determined by evaluation network
        """
        if random.random() < self.epsilon:
            # Exploration: Return index of chosen action within action space. Actual action is self.action_space[<returned_value>]
            return random.randint(0, self.num_actions - 1)

        # Exploitation: Return index of action with highest Q value at current state, as determined by evaluation network
        return np.argmax(self.actor_policy.predict(current_state))

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)
        if self.steps >= STEPS_BEFORE_EPSILON_DECAY:
            self.epsilon *= 0.99999

    def learn(self):
        episode = self.memory.get_episode()
        num_steps = len(episode)
        cumulative_future_rewards = []

        rewards = [e[0] for e in episode]
        log_probabilities = [e[1] for e in episode]

        for t in range(num_steps):
            cumulative_future_reward = 0
            for step, reward in enumerate(rewards):
                cumulative_future_reward += DISCOUNT_FACTOR**step * reward
            cumulative_future_rewards.append(cumulative_future_reward)

        std_cumulative_future_rewards = np.std(cumulative_future_rewards)
        if std_cumulative_future_rewards > 0:
            cumulative_future_rewards = (cumulative_future_rewards - np.mean(cumulative_future_rewards))/std_cumulative_future_rewards

        gradients = [-log_probability*cumulative_future_reward for log_probability, cumulative_future_reward in zip(log_probabilities, cumulative_future_rewards)]

        self.actor_policy.optimise(gradients)