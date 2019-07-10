'''
Any reinforcement library consists of 4 main submodules -

1. *Environment* -This module has various environments.
We would be using any environment that has openAI gym compatible API

2. *Agents* - Various RL algorithms chiefly differ in the way agent records/uses/learns from state/reward information.
This is why every algorithm has a separate agent class. The class will be named after the algorithm.
Some other libraries group common agent API into an abstract base class. I have left out that technical part

3. *Policies/Value/Q functions* - We will be working with realistic environments and therefore tabular policies with
Dynamic programming based learning won't work in our case. Most of our policies will be DNN based.
Policies differ in the way the accept input and what output they produce. But every policy has the same API.
Therefore it is a good idea to have a ABC base policy from which different policies
like CNNPolicy, StackedFrameCNN Policy etc inherit

4. *Utilities* - pre-processing, memory replay buffers, frame stacking buffers etc go here

Our objective is to create a library which lets users integrate any RL algorithm quickly with their own environment.
A sample user code might look something like below (have a glance at the structure and come back).
To achieve that requires consistent API amongst all environments
and agents. We will first go over the API required of Environment, Agent and Policy classes

Navigate to core/agents/base.py
'''

# import required modules
import gym
from agents.a2c import FrameSubstractingAgent
from utils.buffers import FrameSubtractor
from utils.preprocessing import downscale
import random
import numpy as np

# Create environment
env = gym.make('Pong-v0', frameskip=2)

# Obtain State and Action spaces specific to the environment
# Note that following two lines are OpenAI gym environment specific code

NUM_STATES = env.observation_space.shape
# Attribute observation_space returns a Box class instance which has attribute shape

NUM_ACTION_VALUES = 3#env.action_space.n
# NOTE: Atari games are single action which can have multiple values (eg: up=1, down=-1, etc)
# The action space returned is of class Discrete which has a public attribute n
# which tells how many values the action can have. There is no attribute shape on class Discrete (which is inconvinient)
# For more information visit Discrete class documentation at -
# https://github.com/openai/gym/blob/master/gym/spaces/discrete.py

# Set number of episodes to train
NUM_EPISODES = 100000

# Since we have a single discrete action - continuous/discrete observation space problem,
# we can use a simple algorithm like DQN. To see which algorithms require which algorithms, refer -
# https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html
# Create a DQN agent
agent = FrameSubstractingAgent(num_actions=NUM_ACTION_VALUES, observation_space_shape=(84,84), actor_pretrained_policy='Pong-v0.actor.model', critic_pretrained_policy='Pong-v0.critic.model')


# Create a function that runs ONE episode and returns cumulative reward at the end
def run(render=False):
    # Reset the environment to get the initial state. current_state is a single RGB [210 x 160 x 3] image
    current_state = env.reset()
    current_state = downscale(current_state)
    current_state = agent.frame_preprocessor.add_frame(current_state)

    # Set cumulative episode reward to 0
    cumulative_reward = 0

    while True:
        # Based of agent's exploration/exploitation policy, either choose a random action or do a
        # forward pass through agent's policy to obtain action
        current_action = agent.act(current_state)
        action_to_take = current_action+1

        # Take a step in environment
        next_state, reward, is_done, info = env.step(action_to_take)
        next_state = downscale(next_state)
        next_state = agent.frame_preprocessor.add_frame(next_state)

        # NOTE: Remember to reset the frame stacker buffer when episode ends
        if is_done:
            print("Episode Terminated...")

        # the observe method of an agent adds the experience to a experience replay buffer
        # agent.observe((current_state, current_action, reward, next_state, is_done))

        # the learn method -
        # 1. samples uniformly random batch of experience from replay buffer
        # 2. perform one step of mini batch SGD on the policy


        # The stacked next state becomes stacked current state and loop continues
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
            # agent.learn()
            # agent.actor_policy.probabilities.save("Pong-v0.actor.model")
            # agent.critic_policy.value.save("Pong-v0.critic.model")
            agent.frame_preprocessor.reset()
            # agent.memory.reset()
            return cumulative_reward

print("Training Starts..")

# Training code
ep = 0
episode_rewards = []
while ep < NUM_EPISODES:
    episode_reward = run(render=True)
    episode_rewards.append(episode_reward)
    with open("episode_rewards.txt", "w") as f:
        f.write(str(episode_rewards))
    print("Episode {} Terminated with reward {}..".format(ep, episode_reward))
    ep += 1
    if ep > 101:
        avg_reward = np.mean(episode_rewards[-100:])
        print("Episode {}: Average Reward in past 100 eisodes {}, Epsilon: {}, Stes: {}".format(ep, np.mean(episode_rewards[-100:]),
                                                                                                agent.epsilon,
                                                                                                agent.steps))
        if avg_reward > 20:
            break
