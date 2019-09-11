import gym
from agents.REINFORCE.reinforce import REINFORCEAgent
from utils.preprocessing import downscale
import numpy as np
from agents.REINFORCE.hyperparams import BATCH_SIZE, CKPT_PATH

from external_environments.navigate_in_traffic.rl.environment import IRTrafficEnv
from external_environments.navigate_in_traffic.simulator.utils.constants import CAR_HARD_BRAKE, CAR_SMOOTH_BRAKE, CAR_MAX_ACCELERATION


# Create environment
ENV_NAME = 'IRTrafficEnv'
env = IRTrafficEnv(episode_len=500)
action_space = [-CAR_HARD_BRAKE, -CAR_SMOOTH_BRAKE]+list(range(CAR_MAX_ACCELERATION))

NUM_STATES = (84, 84)  # env.observation_space.shape

NUM_ACTION_VALUES = 3  # env.action_space.n

MAX_NUM_EPISODES = 100000

agent = REINFORCEAgent(num_actions=len(action_space), observation_space_shape=(3,),
                       actor_pretrained_policy=None)


# Create a function that runs ONE episode and returns cumulative reward at the end
def run(render=False):
    # Reset the environment to get the initial state. current_state is a single RGB [210 x 160 x 3] image
    current_state = env.reset(render=render)

    # Set cumulative episode reward to 0
    cumulative_reward = 0

    while True:
        # Based of agent's exploration/exploitation policy, either choose a random action or do a
        # forward pass through agent's policy to obtain action
        current_action = agent.act(np.array([current_state]))
        action_to_take = action_space[current_action[0]]

        # Take a step in environment
        next_state, reward, is_done, info = env.step(action_to_take)

        agent.observe((np.array([current_state]), current_action, reward, is_done))

        current_state = next_state

        # Add reward to cumulative episode reward
        cumulative_reward += reward

        if is_done:
            agent.learn()

        # Save policy after every episode and return cumulative earned reward.
        # Note that the saving part is the only CNTK specific code in this entire file
        # Ensuring such modularities are key to building complex libraries
        if is_done:
            if ep % 20 == 0:
                agent.actor_policy.probabilities.save(CKPT_PATH + ENV_NAME + ".actor.ep_{}.model".format(ep))
            agent.memory.reset()
            return cumulative_reward


print("Training Starts..")

# Training code
ep = 1500
episode_rewards = []
while ep < MAX_NUM_EPISODES:
    render = ep % 30 == 0
    episode_reward = run(render=render)
    episode_rewards.append(episode_reward)
    with open(CKPT_PATH + "episode_rewards.txt", "w") as f:
        f.write(str(episode_rewards))
    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(("Episode {}: Reward: {}, Average Reward in past " + str(min(100, len(episode_rewards))) +
          " episodes {}, Steps: {}").format(ep, episode_reward, avg_reward, agent.steps))
    ep += 1

    if avg_reward > 20:
        break
