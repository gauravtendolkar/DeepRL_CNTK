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
from agents.a2c import Agent
from utils.preprocessing import downscale
import random

# Create environment
env = gym.make('Pong-v0', frameskip=5)

# Obtain State and Action spaces specific to the environment
# Note that following two lines are OpenAI gym environment specific code

NUM_STATES = env.observation_space.shape
# Attribute observation_space returns a Box class instance which has attribute shape

NUM_ACTION_VALUES = env.action_space.n
# NOTE: Atari games are single action which can have multiple values (eg: up=1, down=-1, etc)
# The action space returned is of class Discrete which has a public attribute n
# which tells how many values the action can have. There is no attribute shape on class Discrete (which is inconvinient)
# For more information visit Discrete class documentation at -
# https://github.com/openai/gym/blob/master/gym/spaces/discrete.py

# Set number of episodes to train
NUM_EPISODES = 10000

# Since we have a single discrete action - continuous/discrete observation space problem,
# we can use a simple algorithm like DQN. To see which algorithms require which algorithms, refer -
# https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html
# Create a DQN agent
agent = Agent(num_actions=NUM_ACTION_VALUES, observation_space_shape=(84, 84), pretrained_policy=None, replace_target=10)


# Create a function that runs ONE episode and returns cumulative reward at the end
def run(render=False):
    # Reset the environment to get the initial state. current_state is a single RGB [210 x 160 x 3] image
    current_state = env.reset()
    # 3 RGB channels are excess information. The environment has multicolor stuff in it, so to use just one channel,
    # we could just take the red channel. Remember that this does not work for all environments
    # (for example, environment containing pure blue and green elements would be indistinguishable).
    # It is generally a good idea to convert to YCbCr and use luma component of the image as a single channel image

    # We also downscale image to 84x84 pixels. (Generally OK for most Atari games)
    # It is a good idea to visualise current_state after all the pre processing just to check if it makes sense
    # Downscaled, luma channel image -
    # https://github.com/codetendolkar/DeepRL_CNTK/blob/master/core/media/downscaled_image.PNG
    current_state = downscale(current_state)

    # IMPORTANT NOTE: A simgle image does not give enough information to the agent. For example, if the ball is in
    # center of screen in image, the pong agent does not know whether it is going up or down.
    # Information from multiple frames is necessary to play effectively. For some games like tic-tac-toe,
    # multiple frames is not necessary. Multi frame information can be obtained by either stitching
    # multiple sequential frames together or just using multiple sequential frames as a multichannel image.
    # We will use the latter.

    # Remember that Policy architecture depends on its inputs. Therefore in this case we have to
    # choose StackedFrameCNNPolicy (will be described later) which accepts stacked sequential frames as input.
    # The policy has a buffer of stack size where we keep adding states of environment using add_frame method
    stacked_current_state = agent.frame_stacker.add_frame(current_state)

    # Set cumulative episode reward to 0
    cumulative_reward = 0

    while True:
        print("Episode {}, Step {}, Cumulative Reward: {}".format(ep, agent.steps, cumulative_reward))
        # Based of agent's exploration/exploitation policy, either choose a random action or do a
        # forward pass through agent's policy to obtain action
        current_action = agent.act(stacked_current_state)


        # Take a step in environment
        next_state, reward, is_done, info = env.step(current_action)

        # next_state returned by environment is again single RGB [210 x 160 x 3] image
        # so we downscale it and use stack containing previous three frames and
        # downscaled next_state as stacked_next_state
        next_state = downscale(next_state)
        stacked_next_state = agent.frame_stacker.add_frame(next_state)

        # NOTE: Remember to reset the frame stacker buffer when episode ends
        if is_done:
            print("Episode Terminated...")
            agent.frame_stacker.reset()

        # the observe method of an agent adds the experience to a experience replay buffer
        agent.observe((stacked_current_state, current_action, reward, stacked_next_state, is_done))

        # the learn method -
        # 1. samples uniformly random batch of experience from replay buffer
        # 2. perform one step of mini batch SGD on the policy
        agent.learn()

        # The stacked next state becomes stacked current state and loop continues
        stacked_current_state = stacked_next_state

        # Add reward to cumulative episode reward
        cumulative_reward += reward

        # OPTIONAL (slows down training): render method displays the current state of the environment
        if render:
            env.render()

        # Save policy after every episode and return cumulative earned reward.
        # Note that the saving part is the only CNTK specific code in this entire file
        # Ensuring such modularities are key to building complex libraries
        if is_done:
            agent.actor_policy.probabilities.save("pong_actor_ep_{}.model".format(ep))
            agent.critic_policy.value.save("pong_critic_ep_{}.model".format(ep))
            return cumulative_reward


# NOTE: Before we start training, we need to fill the buffer to sample from.
# So we take random actions till fill buffer, but not do a gradient descent pass
current_state = env.reset()
current_state = downscale(current_state)
stacked_current_state = agent.frame_stacker.add_frame(current_state)

print("Filling memory...")

while not agent.memory.is_full():
    # Take random action
    current_action = random.randint(0, agent.num_actions-1)
    # Take step
    next_state, reward, is_done, info = env.step(current_action)
    # Downscale and add to stack
    next_state = downscale(next_state)
    stacked_next_state = agent.frame_stacker.add_frame(next_state)

    if is_done:
        # Reset stack, reset environment
        agent.frame_stacker.reset()
        next_state = env.reset()
        next_state = downscale(next_state)
        stacked_next_state = agent.frame_stacker.add_frame(next_state)

    # add experience to replay buffer
    agent.observe((stacked_current_state, current_action, reward, stacked_next_state, is_done))

print("Training Starts..")

# Training code
ep = avg_reward = 0
while ep < NUM_EPISODES:
    episode_reward = run(render=True)
    print("Episode Terminated..")
    avg_reward = (avg_reward*ep + episode_reward)/(ep+1)
    ep += 1
    print("Eisode {}: Average Reward {}, Epsilon: {}".format(ep, avg_reward, agent.epsilon))
