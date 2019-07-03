import random
import numpy as np
from policies.cnn_policies import CriticStackedFrameCNNPolicy, ActorStackedFrameCNNPolicy, ActorCNNPolicy, CriticCNNPolicy
from policies.nn_policies import CriticNNPolicy, ActorNNPolicy
from utils.buffers import SimpleReplayBuffer, FrameStacker, FrameSubtractor
import warnings

warnings.filterwarnings("ignore")

MAX_EPSILON = 0.1
REPLAY_BUFFER_CAPACITY = 50000
STEPS_BEFORE_EPSILON_DECAY = 1000
BATCH_SIZE = 32
TERMINAL_STATE = None
DISCOUNT_FACTOR = 0.99


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, num_actions, observation_space_shape, pretrained_policy=None, num_frames_to_stack=4, *args, **kwargs):
        self.actor_policy = ActorStackedFrameCNNPolicy(name='Actor Network', num_frames_to_stack=4,
                                                       observation_space_shape=observation_space_shape,
                                                       num_actions=num_actions, pretrained_policy=pretrained_policy)
        self.critic_policy = CriticStackedFrameCNNPolicy(name='Critic Network', num_frames_to_stack=4,
                                                         observation_space_shape=observation_space_shape,
                                                         num_actions=num_actions, pretrained_policy=pretrained_policy)

        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        self.memory = SimpleReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.frame_stacker = FrameStacker(stack_size=num_frames_to_stack, frame_shape=observation_space_shape)

    def act(self, current_state):
        """
        Decide action to take based on current state and exploration strategy. Initailly we explore (i.e. choose uniformly random action) 100% of the times
        But epsilon decays as we gain experience and we start taking policy determined actions, i.e. the action with highest Q value at current state, as determined by evaluation network
        """
        # Exploitation: Return index of action with highest Q value at current state, as determined by evaluation network
        probabilities = np.squeeze(self.actor_policy.predict(current_state))
        print(list(probabilities))
        return np.random.choice(range(self.num_actions), 1, p=probabilities)

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

    def learn(self):
        batch = self.memory.sample(BATCH_SIZE)

        current_states = [e[0] for e in batch]
        actions = np.array([[e[1]] for e in batch], dtype=np.float32)
        next_states = [e[3] for e in batch]

        predicted_current_state_values = self.critic_policy.predict(current_states)
        predicted_next_state_values = self.critic_policy.predict(next_states)

        target_values = []
        td_errors = []
        for i in range(BATCH_SIZE):
            current_state, current_action, reward, next_state, is_done = batch[i]

            if is_done:
                target_values.append([reward])
            else:
                target_values.append(reward + DISCOUNT_FACTOR * predicted_next_state_values[i])

            td_errors.append([target_values[i] - predicted_current_state_values[i]])
        target_values = np.array(target_values, dtype=np.float32)
        td_errors = np.array(td_errors, dtype=np.float32)

        self.actor_policy.optimise(current_states, td_errors, actions)
        self.critic_policy.optimise(current_states, target_values)


class RAMAgent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, num_actions, observation_space_shape, actor_pretrained_policy=None, critic_pretrained_policy=None, *args, **kwargs):
        self.actor_policy = ActorNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                                       num_actions=num_actions, pretrained_policy=actor_pretrained_policy)
        self.critic_policy = CriticNNPolicy(name='Critic Network', observation_space_shape=observation_space_shape,
                                                         num_actions=num_actions, pretrained_policy=critic_pretrained_policy)

        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        self.memory = SimpleReplayBuffer(REPLAY_BUFFER_CAPACITY)

    def act(self, current_state):
        """
        Decide action to take based on current state and exploration strategy. Initailly we explore (i.e. choose uniformly random action) 100% of the times
        But epsilon decays as we gain experience and we start taking policy determined actions, i.e. the action with highest Q value at current state, as determined by evaluation network
        """
        # Exploitation: Return index of action with highest Q value at current state, as determined by evaluation network
        probabilities = np.squeeze(self.actor_policy.predict(current_state))
        return np.random.choice(range(self.num_actions), 1, p=probabilities)[0]

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

    def learn(self):
        batch = self.memory.sample(BATCH_SIZE)

        current_states = [e[0] for e in batch]
        actions = np.array([[e[1]] for e in batch], dtype=np.float32)
        next_states = [e[3] for e in batch]

        predicted_current_state_values = self.critic_policy.predict(current_states)
        predicted_next_state_values = self.critic_policy.predict(next_states)
        target_values = []
        for i in range(BATCH_SIZE):
            current_state, current_action, reward, next_state, is_done = batch[i]

            if is_done:
                target_values.append([reward])
            else:
                target_values.append(reward + DISCOUNT_FACTOR * predicted_next_state_values[i])

        target_values = np.array(target_values, dtype=np.float32).squeeze()
        target_values = np.divide((target_values - np.mean(target_values)), np.std(target_values))
        target_values = target_values.reshape((-1, 1))

        td_errors = target_values - predicted_current_state_values
        td_errors = np.array(td_errors, dtype=np.float32)

        self.actor_policy.optimise(current_states, td_errors, actions)
        self.critic_policy.optimise(current_states, target_values)


class FrameSubstractingAgent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, num_actions, observation_space_shape, actor_pretrained_policy=None, critic_pretrained_policy=None, *args, **kwargs):
        self.actor_policy = ActorCNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                           num_actions=num_actions, pretrained_policy=actor_pretrained_policy)
        self.critic_policy = CriticCNNPolicy(name='Critic Network', observation_space_shape=observation_space_shape,
                                             num_actions=num_actions, pretrained_policy=critic_pretrained_policy)

        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        self.memory = SimpleReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.frame_preprocessor = FrameSubtractor()

    def act(self, current_state):
        """
        Decide action to take based on current state and exploration strategy. Initailly we explore (i.e. choose uniformly random action) 100% of the times
        But epsilon decays as we gain experience and we start taking policy determined actions, i.e. the action with highest Q value at current state, as determined by evaluation network
        """
        # Exploitation: Return index of action with highest Q value at current state, as determined by evaluation network
        current_state = self.frame_preprocessor.add_frame(current_state)
        probabilities = np.squeeze(self.actor_policy.predict(current_state))
        return np.random.choice(range(self.num_actions), 1, p=probabilities)

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

    def learn(self):
        batch = self.memory.sample(BATCH_SIZE)

        current_states = [e[0] for e in batch]
        actions = np.array([[e[1]] for e in batch], dtype=np.float32)
        next_states = [e[3] for e in batch]

        predicted_current_state_values = self.critic_policy.predict(current_states)
        predicted_next_state_values = self.critic_policy.predict(next_states)

        target_values = []
        for i in range(BATCH_SIZE):
            current_state, current_action, reward, next_state, is_done = batch[i]

            if is_done:
                target_values.append([reward])
            else:
                target_values.append(reward + DISCOUNT_FACTOR * predicted_next_state_values[i])

        target_values = np.array(target_values, dtype=np.float32).squeeze()
        target_values = np.divide((target_values - np.mean(target_values)), np.std(target_values))
        target_values = target_values.reshape((-1, 1))

        td_errors = target_values - predicted_current_state_values
        td_errors = np.array(td_errors, dtype=np.float32)

        self.actor_policy.optimise(current_states, td_errors, actions)
        self.critic_policy.optimise(current_states, target_values)
