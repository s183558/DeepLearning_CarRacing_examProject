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
episodes = 10
batch_size = 32
step_size = batch_size * 4
verbose_freq = 50

figure_file = 'plots/CarRacing_rewards.png'

######################
#     Environment    #
######################
env = gym.make('CarRacing-v2', new_step_api=True) # Create environment
#env = Recorder(env, "./video", auto_release = False) # To display environment in Colab
env.reset(options={"randomize": False}, seed = 42) # Reset environment
env = wrappers.GrayScaleObservation(env, keep_dim = True)

obs_shape = env.observation_space.shape

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
        if action[1] > 0.1:
            bonus_reward = abs(reward *0.5)
            
        # Add the reward from the step to our score
        score += reward + bonus_reward
        
        # Throm all our variables in the memory
        agent.remember(state, action, reward, state_next, np.array([done, done2]).any())
        
        # When we have enough state-action pairs, we can update our online nets
        if len(agent.memory) > batch_size:
            agent.update()

        # The next state, is now our current state.
        state = state_next
        
        # if the environment terminates before the step_size, we break
        if np.array([done, done2]).any():
            print(f'Terminated at:\nEpisode: {e+1}, reward: {score:.1f}, avg_reward: {np.mean(rewards[-100:]):.1f}')
            break
        
        # How far into the episode we are 
        if (step+1) % verbose_freq == 0: print(f'Step {step+1} done!')
            
    # After each episode we store the score        
    rewards.append(score)
    
    # The avg score of the last 100 episodes
    avg_score = np.mean(rewards[-100:])
    
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
    
    print(f'Episode {e+1}, score: {score:.1f}, avg_score: {avg_score:.1f}')

# Save the final paramters of the model
agent.save_models('final')

x = [i + 1 for i in range(episodes)]
plot_learning_curve(x, rewards, figure_file)