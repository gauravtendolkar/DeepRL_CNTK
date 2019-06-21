# Create environment
import gym
from preprocessing import downscale
import random
env = gym.make('Pong-v0', frameskip=5)
env.reset()

# State and Action spaces
NUM_STATES = env.observation_space.shape
NUM_ACTION_VALUES = env.action_space.n

NUM_EPISODES = 10000

# Create Agent
from dqn_agent import Agent
agent = Agent(num_actions=NUM_ACTION_VALUES, observation_space_shape=(84, 84))


def run():
    current_state = env.reset()
    current_state = downscale(current_state)
    cumulative_reward = 0

    while True:
        current_action = agent.act(current_state)

        next_state, reward, is_done, info = env.step(current_action)
        next_state = downscale(next_state)

        if is_done:
            print("Episode Terminated...")
            next_state = None

        agent.observe((current_state, current_action, reward, next_state))
        agent.learn()

        current_state = next_state
        cumulative_reward += reward

        env.render()

        if is_done:
            agent.evaluation_network.q.save("crosser.model")
            return cumulative_reward

current_state = env.reset()
current_state = downscale(current_state)

print("Filling memory...")

while not agent.memory.is_full():
    current_action = random.randint(0, agent.num_actions-1)
    next_state, reward, is_done, info = env.step(current_action)
    next_state = downscale(next_state)
    if is_done:
        next_state = env.reset()
        next_state = downscale(next_state)
    agent.observe((current_state, current_action, reward, next_state))
    current_state = next_state

print("Done..")

ep = avg_reward = 0
while ep < NUM_EPISODES:
    episode_reward = run()
    print("Episode Terminated..")
    avg_reward = (avg_reward*ep + episode_reward)/(ep+1)
    ep += 1
    print("Eisode {}: Average Reward {}, Epsilon: {}".format(ep, avg_reward, agent.epsilon))
