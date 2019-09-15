# import required modules
import gym
import random
import numpy as np
from agents.dqn.dqn import RAMAgent

# Create environment
# In this case we are using a open AI gyn registered environment
# To use a custom environment, we might use something like - ```env = MyCustomEnv()```
env = gym.make('LunarLander-v2')

# Obtain State and Action spaces specific to the environment
# Note that following two lines are OpenAI gym environment specific code

NUM_STATES = env.observation_space.shape
# Attribute observation_space returns a Box class instance which has attribute shape

NUM_ACTION_VALUES = env.action_space.n
# NOTE: Atari games are single action which can have multiple values (eg: up=1, down=-1, etc)
# The action space returned is of class Discrete which has a public attribute n
# which tells how many values the action can have. There is no attribute shape on class Discrete
# For more information visit Discrete class documentation at -
# https://github.com/openai/gym/blob/master/gym/spaces/discrete.py

# Set number of maximum episodes to train
NUM_EPISODES = 10000

# Since we have a single discrete action and continuous/discrete observation space problem,
# we can use a simple algorithm like DQN. To see which algorithms require which algorithms, refer -
# https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html
# Create a DQN agent

# Although majority of code is shared between RAMAgent and PixelAgent, we maintain 2 separate classes for understanding.
# A better implementation would abstract out the common code into a base class
agent = RAMAgent(num_actions=NUM_ACTION_VALUES,
                 observation_space_shape=NUM_STATES,
                 pretrained_policy=None,
                 replace_target=1000)


# Create a function that runs ONE episode and returns cumulative reward at the end
def run(render=False):

    # Reset the environment to get the initial state. current_state is a single list of 8 32-bit floating point numbers
    # These numbers completely encapsulate the state of the game.
    # Note that the 8 numbers are just a way to represent 32*8 bits. The 32*8 bits contain the state of game
    current_state = env.reset()

    # Since state is a list, we convert it to a Numpy array for computational efficiency
    current_state = np.array(current_state, dtype=np.float32)

    # Set cumulative episode reward to 0. cumulative episode reward tracks total return of an episode
    cumulative_reward = 0

    while True:
        # forward pass through agent's policy to obtain action
        current_action = agent.act(current_state)

        # It could be the case that the environment accepts actions -1,0,1 but the policy produces output 0,1,2
        # In that case you may add a code line that translates current_action to action within action space
        # In current case action_to_take is equal to current_action
        action_to_take = current_action

        # Take a step in environment
        next_state, reward, is_done, info = env.step(action_to_take)

        # Convert the next_state list to array
        next_state = np.array(next_state, dtype=np.float32)

        # NOTE: You may want to perform other actions here like reset the preprocessor states etc
        # when episode ends
        if is_done:
            print("Episode Terminated...")

        # the observe method of an agent adds the experience to a experience replay buffer
        agent.observe((current_state, current_action, reward, next_state, is_done))

        # the learn method -
        # 1. samples uniformly random batch of experience from replay buffer
        # 2. perform one step of mini batch SGD on the policy
        agent.learn()

        # The next state becomes current state and loop continues
        current_state = next_state

        # Add reward to cumulative episode reward
        cumulative_reward += reward

        # OPTIONAL (slows down training): render method displays the current state of the environment
        if render:
            env.render()

        # Save policy after every episode and return cumulative earned reward.
        # Note that the saving part is the only CNTK specific code in this entire file
        # Ensuring such modularities are key to building complex libraries
        if is_done:
            agent.evaluation_policy.q.save("LunarLander.model")
            return cumulative_reward


# NOTE: Before we start training, we need to fill the buffer to sample from.
# So we take random actions till full buffer, but not do a gradient descent pass
current_state = env.reset()
current_state = np.array(current_state, dtype=np.float32)

print("Filling memory...")

while not agent.memory.is_full():
    # Take random action
    current_action = random.randint(0, agent.num_actions-1)
    action_to_take = current_action

    # Take step
    next_state, reward, is_done, info = env.step(action_to_take)
    next_state = np.array(next_state, dtype=np.float32)

    if is_done:
        # Reset stack, reset environment
        next_state = env.reset()
        next_state = np.array(next_state, dtype=np.float32)

    # add experience to replay buffer
    agent.observe((current_state, current_action, reward, next_state, is_done))

    # The next state becomes current state and loop continues
    current_state = next_state

print("Training Starts..")

# Training code
ep = avg_reward = 0

while ep < NUM_EPISODES:
    episode_reward = run(render=False)

    # Calculate running average of past 100 episodes
    if ep % 100 == 0:
        avg_reward = 0
    avg_reward = (avg_reward * (ep % 100) + episode_reward) / ((ep % 100) + 1)

    ep += 1
    print("Episode {}: Average Reward in past 100 eisodes {}, Epsilon: {}, Stes: {}".format(ep, avg_reward, agent.epsilon, agent.steps))
