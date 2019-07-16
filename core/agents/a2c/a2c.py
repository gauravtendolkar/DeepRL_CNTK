import numpy as np
from policies.cnn_policies_2 import CriticStackedFrameCNNPolicy, ActorStackedFrameCNNPolicy, ActorCNNPolicy, CriticCNNPolicy
from policies.nn_policies import CriticNNPolicy, ActorNNPolicy
from utils.buffers import SimpleBuffer, FrameStacker, FrameSubtractor
from agents.a2c.hyperparams import BATCH_SIZE, DISCOUNT_FACTOR


class A2CAgent:
    steps = 0

    def __init__(self, num_actions, observation_space_shape, actor_pretrained_policy=None, critic_pretrained_policy=None, *args, **kwargs):
        self.actor_policy = ActorCNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                           num_actions=num_actions, pretrained_policy=actor_pretrained_policy)
        self.critic_policy = CriticCNNPolicy(name='Critic Network', observation_space_shape=observation_space_shape,
                                             num_actions=num_actions, pretrained_policy=critic_pretrained_policy)

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
        batch = self.memory.get_data()
        n = len(batch)

        current_states = [e[0] for e in batch]
        actions = [[e[1]] for e in batch]
        rewards = [[e[2]] for e in batch]
        next_states = [e[3] for e in batch]

        predicted_current_state_values = self.critic_policy.predict(current_states)
        predicted_next_state_values = self.critic_policy.predict(next_states)
        target_next_state_values = rewards + DISCOUNT_FACTOR*predicted_next_state_values
        td_0s = target_next_state_values - predicted_current_state_values

        for i in range((n // BATCH_SIZE) + 1):
            start, end = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, n)
            if start < end:
                self.actor_policy.optimise(current_states[start:end], td_0s[start:end], actions[start:end])
                self.critic_policy.optimise(current_states[start:end], target_next_state_values[start:end])

        self.memory.reset()


class A2CSharedCNNAgent:
    steps = 0

    def __init__(self, num_actions, observation_space_shape, actor_pretrained_policy=None,
                 critic_pretrained_policy=None, shared_cnn=False, *args, **kwargs):

        if shared_cnn:
            self.policy = ActorCriticCNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                               num_actions=num_actions, pretrained_policy=actor_pretrained_policy)
        else:
            self.actor_policy = ActorCNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                               num_actions=num_actions, pretrained_policy=actor_pretrained_policy)
            self.critic_policy = CriticCNNPolicy(name='Critic Network', observation_space_shape=observation_space_shape,
                                                 num_actions=num_actions, pretrained_policy=critic_pretrained_policy)

        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        # self.memory = EpisodicBuffer()
        self.frame_preprocessor = FrameSubtractor()

    def act(self, current_state):
        """
        Decide action to take based on current state and exploration strategy. Initailly we explore (i.e. choose uniformly random action) 100% of the times
        But epsilon decays as we gain experience and we start taking policy determined actions, i.e. the action with highest Q value at current state, as determined by evaluation network
        """
        # Exploitation: Return index of action with highest Q value at current state, as determined by evaluation network
        probabilities = np.squeeze(self.actor_policy.predict(current_state))
        action = np.random.choice(range(self.num_actions), 1, p=probabilities)
        return action

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

    def learn(self):
        episode = self.memory.get_episode()
        n = len(episode)

        current_states = np.array([e[0] for e in episode], dtype=np.float32)
        actions = np.array([[e[1]] for e in episode], dtype=np.float32)
        rewards = np.array([e[2] for e in episode], dtype=np.float32)
        next_states = np.array([e[3] for e in episode], dtype=np.float32)

        predicted_current_state_values = self.critic_policy.predict(current_states)

        returns = np.zeros(n, dtype=np.float32)
        for i in range(n):
            ret = 0
            for t in range(i, n):
                ret += (DISCOUNT_FACTOR ** (t - i)) * rewards[t]
            returns[i] = ret

        returns = np.divide((returns - np.mean(returns)), np.std(returns) + 0.000001)
        returns = returns.reshape((-1, 1))

        td_errors = returns - predicted_current_state_values
        td_errors = np.array(td_errors, dtype=np.float32)

        shuffled_idx = list(range(n))
        np.random.shuffle(shuffled_idx)

        for i in range((n // BATCH_SIZE) + 1):
            if i * BATCH_SIZE < min((i + 1) * BATCH_SIZE, n):
                idx = shuffled_idx[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, n)]
                self.actor_policy.optimise(current_states[idx], td_errors[idx], actions[idx])
                self.critic_policy.optimise(current_states[idx], returns[idx])


class A2CMultiEnvSharedCNNAgent:
    steps = 0

    def __init__(self, num_actions, observation_space_shape, actor_pretrained_policy=None,
                 critic_pretrained_policy=None, shared_cnn=False, *args, **kwargs):

        if shared_cnn:
            self.policy = ActorCriticCNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                               num_actions=num_actions, pretrained_policy=actor_pretrained_policy)
        else:
            self.actor_policy = ActorCNNPolicy(name='Actor Network', observation_space_shape=observation_space_shape,
                                               num_actions=num_actions, pretrained_policy=actor_pretrained_policy)
            self.critic_policy = CriticCNNPolicy(name='Critic Network', observation_space_shape=observation_space_shape,
                                                 num_actions=num_actions, pretrained_policy=critic_pretrained_policy)

        self.num_actions = num_actions
        self.observation_space_shape = observation_space_shape
        # self.memory = EpisodicBuffer()
        self.frame_preprocessor = FrameSubtractor()

    def act(self, current_state):
        """
        Decide action to take based on current state and exploration strategy. Initailly we explore (i.e. choose uniformly random action) 100% of the times
        But epsilon decays as we gain experience and we start taking policy determined actions, i.e. the action with highest Q value at current state, as determined by evaluation network
        """
        # Exploitation: Return index of action with highest Q value at current state, as determined by evaluation network
        probabilities = np.squeeze(self.actor_policy.predict(current_state))
        action = np.random.choice(range(self.num_actions), 1, p=probabilities)
        return action

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

    def learn(self):
        episode = self.memory.get_episode()
        n = len(episode)

        current_states = np.array([e[0] for e in episode], dtype=np.float32)
        actions = np.array([[e[1]] for e in episode], dtype=np.float32)
        rewards = np.array([e[2] for e in episode], dtype=np.float32)
        next_states = np.array([e[3] for e in episode], dtype=np.float32)

        predicted_current_state_values = self.critic_policy.predict(current_states)

        returns = np.zeros(n, dtype=np.float32)
        for i in range(n):
            ret = 0
            for t in range(i, n):
                ret += (DISCOUNT_FACTOR ** (t - i)) * rewards[t]
            returns[i] = ret

        returns = np.divide((returns - np.mean(returns)), np.std(returns) + 0.000001)
        returns = returns.reshape((-1, 1))

        td_errors = returns - predicted_current_state_values
        td_errors = np.array(td_errors, dtype=np.float32)

        shuffled_idx = list(range(n))
        np.random.shuffle(shuffled_idx)

        for i in range((n // BATCH_SIZE) + 1):
            if i * BATCH_SIZE < min((i + 1) * BATCH_SIZE, n):
                idx = shuffled_idx[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, n)]
                self.actor_policy.optimise(current_states[idx], td_errors[idx], actions[idx])
                self.critic_policy.optimise(current_states[idx], returns[idx])