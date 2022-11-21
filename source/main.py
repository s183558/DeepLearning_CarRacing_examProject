import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

from agent import DDPGAgent

######################
#     Parameters     #
######################
gamma = 0.9
tau = 0.01
lr_P, lr_Q = 0.0001, 0.0001
episodes = 10
batch_size = 32
step_size = batch_size * 4
verbose_freq = 10

######################
#     Environment    #
######################
env = gym.make('CarRacing-v2', new_step_api=True) # Create environment
#env = Recorder(env, "./video", auto_release = False) # To display environment in Colab
env.reset(options={"randomize": False}, seed = 42) # Reset environment
env = wrappers.GrayScaleObservation(env)

obs_shape = env.observation_space.shape

# Agent
agent = DDPGAgent(gamma, tau, lr_P, lr_Q)

# Training
rewards = []

for e in range(episodes):

    state = env.reset()   # (96, 96) image
    reward_episode = 0

    for step in range(step_size):
        #plt.imshow(state, cmap="gray")
        action = agent.getAction(state)[0]
        
        state_next, reward, done, done2, _ = env.step(action)
        agent.memory.push(state, action, reward, state_next, np.array([done, done2]).any())

        if len(agent.memory) > batch_size:
            agent.update(batch_size)
            agent.update(batch_size)
            break

        state = state_next
        reward_episode += reward

        if np.array([done, done2]).any():
            print("episode: {:2}, reward: {:.2f}, average _reward: {}".format(
                    e, reward_episode, np.mean(rewards)))
            break
        if (step+1) % verbose_freq == 0:
            print(f'Step {step} done!')
    print(f'Episode {e+1} done!')
    rewards.append(reward_episode)
    break
