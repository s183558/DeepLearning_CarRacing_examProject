drive_path = ''

import numpy as np
import gym
from gym import wrappers
import wandb
#wandb.login()

from agent import DDPGAgent
from utils import plot_learning_curve

######################
#     Parameters     #
######################
gamma = 0.99
tau = 0.005
lr_mu, lr_Q = 0.0001, 0.0002
episodes = 10
batch_size = 500
step_size = 1000
verbose_freq = 50

figure_file = 'plots/CarRacing_rewards.png'

do_wandb = False

if do_wandb:
    wandb_client = wandb.init(project="DL_CarRacing2")

######################
#     Environment    #
######################
env = gym.make('CarRacing-v2', new_step_api=True) # Create environment
env.reset(options={"randomize": False}, seed = 42) # Reset environment
env = wrappers.GrayScaleObservation(env, keep_dim = True)


######################
#       Agent        #
######################
agent = DDPGAgent(gamma = gamma, tau = tau,lr_mu = lr_mu,
                  lr_Q = lr_Q, env = env, batch_size = batch_size)

print(f'Online {agent.critic}')
print(f'Online {agent.actor}')

#######################
# Training parameters #
#######################
rewards = []
best_score = 0
no_reward_counter = 0

# Wandb variables
actor_loss = [[],[]]
critic_loss = [[],[]]
metrics = []
metric_counter_step = 0
metric_counter_model = 0

#######################
#      Main loop      #
#######################

for e in range(episodes):

    state = env.reset()   # (96, 96, 1) image
    score = 0

    for step in range(step_size):
        # Sample an action from our actor network
        action = agent.getAction(state)[0]
       
        
        # Take the action in our current state
        state_next, reward, done, done2, _ = env.step(action)
        
        env.render()
        
        # If the agent gives some gas it get a bonus reward
        bonus_reward = 0
        if action[1] > 0.1:
            bonus_reward = 0.05 #0.5 + action[1] #abs(reward *0.5)
        
        # Position of the car
        x = env.car.hull.position[0]
        y = env.car.hull.position[1]

        if do_wandb:
            metrics.append({"reward"   : reward,
                            "bonus_reward" : bonus_reward,
                            "Steering" : action[0],
                            "Gas"      : action[1],
                            "Breaking" : action[2],
                            "x pos"    : x,
                            "y pos"    : y,
                           })

        # Add the reward from the step to our score
        reward += bonus_reward
        score += reward
        
        # Throw all our variables in the memory
        agent.remember(state, action, reward, state_next, np.array([done, done2]).any())
        

        # When we have enough state-action pairs, we can update our online nets
        if len(agent.small_memory) == batch_size:
            print('Updating the networks...')
            # Update network with small buffer for recency
            smallB_metrics = agent.update(recency_buffer = True)
            
            # Move small buffer to big, and reset small buffer
            agent.move_small_buffer_to_big()
            
            # Update network with big buffer
            bigB_metrics = agent.update(recency_buffer = False)
            
            if do_wandb:
                for i in range(batch_size):
                    metrics[i].update({
                             "Q-val (short term)": smallB_metrics[2][i],
                             "y_i (short term)"  : smallB_metrics[3][i],
                             "Q-val (long term)" : bigB_metrics[2][i],
                             "y_i (long term)"   : bigB_metrics[3][i]})
                    wandb_client.log(metrics[i])
                metrics = []
            
            critic_loss[0].append(smallB_metrics[0])
            critic_loss[1].append(bigB_metrics[0])
            actor_loss[0].append(smallB_metrics[1])
            actor_loss[1].append(bigB_metrics[1])
            
            
            

        # The next state, is now our current state.
        state = state_next
        
        # How far into the episode we are 
        if (step+1) % verbose_freq == 0: print(f'Step {step+1} done!')
        
        # if the environment terminates before the step_size, we break
        if np.array([done, done2]).any():
            print('## ## Terminated at:')
            break
        
        # If we dont get a positive reward in 200 steps, we reset
        
        if reward < 0:
            no_reward_counter += 1
            if no_reward_counter > 400:
                print('No positive reward in 200 steps; RESET')
                break
        else:
            no_reward_counter = 0
            
        
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

if do_wandb:
    wandb_client.log(
            {"actor_loss" : wandb.plot.line_series(
                           xs = range(len(actor_loss)), 
                           ys = actor_loss,
                           keys = ["Short term", "long term"],
                           title = "Actor loss",
                           xname = "Step")})
    wandb_client.log(
            {"critic_loss" : wandb.plot.line_series(
                           xs = range(len(critic_loss)), 
                           ys = critic_loss,
                           keys = ["Short term", "long term"],
                           title = "Critic loss",
                           xname = "Step")})
    wandb_client.finish()

# Save the final paramters of the model
agent.save_models(suffix = '_final', drive_dir = drive_path)

x = [i + 1 for i in range(len(rewards))]
plot_learning_curve(x, rewards, figure_file)