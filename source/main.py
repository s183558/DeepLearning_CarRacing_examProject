drive_path = ''

import numpy as np
import gym
from gym import wrappers

from agent import DDPGAgent
from utils import plot_learning_curve

######################
#     Parameters     #
######################
gamma = 0.99
tau = 0.005
lr_mu, lr_Q = 0.0001, 0.0002
episodes = 200
batch_size = 500
step_size = 1000
verbose_freq = 50


figure_file = 'plots/CarRacing_rewards.png'

######################
#     Environment    #
######################
env = gym.make('CarRacing-v2', new_step_api=True) # Create environment
env.reset(options={"randomize": False}, seed = 42) # Reset environment
env = wrappers.GrayScaleObservation(env, keep_dim = True)

# Agent
agent = DDPGAgent(gamma = gamma, tau = tau,lr_mu = lr_mu,
                  lr_Q = lr_Q, env = env, batch_size = batch_size)

print(f'Online {agent.critic}')
print(f'Online {agent.actor}')

# Training
rewards = []
best_score = -5000


for e in range(episodes):

    state = env.reset()   # (96, 96, 1) image
    score = 0

    for step in range(step_size):
        # Sample an action from our actor network
        action = agent.getAction(state)[0]
        
        # Take the action in our current state
        state_next, reward, done, done2, _ = env.step(action)
        
        # If the agent gives some gas it get a bonus reward
        bonus_reward = 0
        if action[1] > 0.1:
            bonus_reward = 0.5 + action[1] #abs(reward *0.5)
            
        # Add the reward from the step to our score
        score += reward + bonus_reward
        
        # Throw all our variables in the memory
        agent.remember(state, action, reward, state_next, np.array([done, done2]).any())
        

        # When we have enough state-action pairs, we can update our online nets
        if len(agent.small_memory) == batch_size:
            print(f'Updating the networks...')
            # Update network with small buffer for recency
            agent.update(recency_buffer = True)
            
            # Move small buffer to big, and reset small buffer
            agent.move_small_buffer_to_big()
            
            # Update network with big buffer
            agent.update(recency_buffer = False)
            

        # The next state, is now our current state.
        state = state_next
        
        # How far into the episode we are 
        if (step+1) % verbose_freq == 0: print(f'Step {step+1} done!')
        
        # if the environment terminates before the step_size, we break
        if np.array([done, done2]).any():
            print(f'## ## Terminated at:')
            break
        
    # After each episode we store the score        
    rewards.append(score)
    
    # The avg score of the last 10 episodes
    avg_score = np.mean(rewards[-10:])
    
    print(f'Episode {e+1}, score: {score:.1f}, avg_score: {avg_score:.1f}')
    
    # Save the model, every 100th episode
    if (e+1) % 100 == 0: agent.save_models(suffix = f'_episode_{e+1}',
                                           drive_dir = drive_path)
    
    # Save the best model
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models(drive_dir = drive_path)
    
    

# Save the final paramters of the model
agent.save_models(suffix = '_final', drive_dir = drive_path)

x = [i + 1 for i in range(len(rewards))]
plot_learning_curve(x, rewards, figure_file)