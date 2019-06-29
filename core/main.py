import gym
from agents.dqn import Agent
from utils.preprocessing import downscale
import random

# Create environment
env = gym.make('Pong-ram-v0', frameskip=2)

# Obtain State and Action spaces specific to the environment
# Note that following two lines are OpenAI gym environment specific code

NUM_STATES = env.observation_space.shape
# Attribute observation_space returns a Box class instance which has attribute shape

NUM_ACTION_VALUES = env.action_space.n

print(NUM_STATES, NUM_ACTION_VALUES)
