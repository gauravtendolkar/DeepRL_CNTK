# Create environment
import gym
from preprocessing import downscale
env = gym.make('Freeway-v0')
env.reset()

# State and Action spaces
NUM_STATES = env.observation_space.shape
NUM_ACTION_VALUES = env.action_space.shape

# Create Agent
from dqn_agent import Agent
agent = Agent(num_actions=NUM_ACTION_VALUES, observation_space_shape=NUM_STATES[:2])


def run():
    current_state = env.reset()
    current_state = downscale(current_state)
    cumulative_reward = 0



    while True:
        current_action = agent.act(current_state)
        print("action details: ",current_action, agent.num_actions)

        next_state, reward, is_done, info = env.step(current_action)
        next_state = downscale(next_state)

        if is_done:
            next_state = None

        agent.observe((current_state, action, reward, next_state))
        agent.learn()

        print(action)

        current_state = next_state
        cumulative_reward += reward

        if done:
            return cumulative_reward

