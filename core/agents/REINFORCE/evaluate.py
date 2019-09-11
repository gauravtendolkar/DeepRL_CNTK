import gym
from agents.REINFORCE.reinforce import REINFORCEAgent
from utils.preprocessing import downscale
import numpy as np
from agents.REINFORCE.hyperparams import BATCH_SIZE, CKPT_PATH

# Create environment
ENV_NAME = 'Pong-v0'
env = gym.make(ENV_NAME, frameskip=2)

NUM_STATES = (84, 84)  # env.observation_space.shape

NUM_ACTION_VALUES = 3  # env.action_space.n

agent = REINFORCEAgent(num_actions=NUM_ACTION_VALUES, observation_space_shape=(84, 84),
                       actor_pretrained_policy='ckpt/Pong-v0.actor.ep_3000.model')


# Create a function that runs ONE episode and returns cumulative reward at the end
def run(render=False):
    # Reset the environment to get the initial state. current_state is a single RGB [210 x 160 x 3] image
    current_state = env.reset()
    current_state = downscale(current_state)
    current_state = agent.frame_preprocessor.add_frame(current_state)

    # Set cumulative episode reward to 0
    cumulative_reward = 0

    while True:
        vr.capture_frame()
        # Based of agent's exploration/exploitation policy, either choose a random action or do a
        # forward pass through agent's policy to obtain action
        current_action = agent.act(current_state)
        action_to_take = current_action + 1

        # Take a step in environment
        next_state, reward, is_done, info = env.step(action_to_take)
        next_state = downscale(next_state)
        next_state = agent.frame_preprocessor.add_frame(next_state)

        current_state = next_state

        # Add reward to cumulative episode reward
        cumulative_reward += reward

        # OPTIONAL (slows down training): render method displays the current state of the environment
        #env.render()


        # Save policy after every episode and return cumulative earned reward.
        # Note that the saving part is the only CNTK specific code in this entire file
        # Ensuring such modularities are key to building complex libraries
        if is_done:
            agent.frame_preprocessor.reset()
            agent.memory.reset()
            return cumulative_reward


print("Training Starts..")

# Training code
vr = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, './pretrained/demo.mp4')
episode_reward = run(render=False)
vr.close()
