import numpy as np
import random
from collections import deque
import torch
import matplotlib.pyplot as plt


class Memory:
    def __init__(self, max_size):
        # Initiate a que for our buffer to be stored in
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        # Store all the values in the buffer
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        # Pick "batch_size" number of elements randomly from the buffer
        # and then return them
        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return np.array(state_batch), np.array(action_batch),\
               np.array(reward_batch), np.array(next_state_batch), \
               np.array(done_batch)

    def __len__(self):
        return len(self.buffer)



def info_about_env(env):
    # The action space 
    a = env.action_space
    print(f'Action space: [Steering, gas, breaking];\
                \n\t\tmin_val = {a.low},\
                \n\t\tmax_val = {a.high},\
                \n\t\tNo_action = {a.shape[0]},\
                \n\t\ttype = {a.dtype}' )

    # The return values of the env.step
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print('What is returned from the environment, for each step taken:')
    print(f'State space, with shape ({type(state).__name__}): {state.shape}')
    print(f'Reward of the action ({type(reward).__name__}): {reward}')
    print(f'Terminated ({type(terminated).__name__}): {terminated}')
    print(f'Truncated ({type(truncated).__name__}): {truncated}')
    print(f'Info ({type(info).__name__}): {info}')


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x, running_avg, label = 'running avg')
    plt.plot(x, scores, label = 'score')
    plt.xlabel = 'Episodes'
    plt.ylabel = 'Reward'
    plt.title('Running avg of last 100 scores')
    plt.legend(loc = 'best')
    plt.savefig(figure_file)


