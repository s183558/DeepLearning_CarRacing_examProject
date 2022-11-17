import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from agent import DDPGAgent
from utils import processState


# Parameters
gamma = 0.9
tau = 0.01
lr_P, lr_Q = 0.0001, 0.0001
episodes = 50
batch_size = 32

# Environment
env = gym.make("CarRacing-v2")

# Agent
agent = DDPGAgent(gamma, tau, lr_P, lr_Q)

# Training
rewards = []

for e in range(episodes):

    state_, _ = env.reset()
    state = processState(state_)
    reward_episode = 0

    while True:
        plt.imshow(state[0, 0, :, :], cmap="gray")
        action = agent.getAction(state)
        state_next_, reward, done, _ = env.step(action)
        state_next = processState(state_next_)
        agent.memory.push(state, action, reward, state_next, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = state_next
        reward_episode += reward

        if done:
            print(
                "episode: {:2}, reward: {:.2f}, average _reward: {}".format(
                    e, reward_episode, np.mean(rewards)
                )
            )
            break

    rewards.append(reward_episode)
