import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import CriticNetwork, ActorNetwork
from utils import Memory

class DDPGAgent:
    def __init__(self, lr_mu, lr_Q, gamma, tau, env, batch_size = 32,
                 noise_std  = [0.1, 0.05, 0.05]):
        
        # Choose the device to run on
        if torch.cuda.is_available():
            print('Agent is running on GPU')
            self.device = torch.device('cuda')
        else:
            print('Agent is running on CPU')
            self.device = torch.device('cpu')
        
        # Parameters
        self.max_memory_size = 256
        self.gamma           = gamma
        self.tau             = tau
        self.batch_size      = batch_size
        
        self.min_action_val = torch.FloatTensor(env.action_space.low).to(self.device)
        self.max_action_val = torch.FloatTensor(env.action_space.high).to(self.device)
        self.noise_std      = torch.FloatTensor(noise_std).to(self.device)
        
        
        # Randomly initialize the critic (Q) and actor (mu) network
        self.critic        = CriticNetwork(name='Critic').to(self.device)
        self.actor         = ActorNetwork(name ='Actor').to(self.device)
        self.target_critic = CriticNetwork(name='TargetCritic').to(self.device)
        self.target_actor  = ActorNetwork(name ='TargetActor').to(self.device)
        
        # Set the target networks to have the save parameters as their online version
        self.update_target_networks(tau = 1)

        # Initialize the replay buffer
        self.memory = Memory(self.max_memory_size)
        
        # Define the loss function and the optimizer
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_Q)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=lr_mu)
        
        
        
    
    def remember(self, state, action, reward, state_next, done):
        # Change the state dimensions to be correct, aka.:(channel, height, width)
        state = np.transpose(state, (2, 0, 1))
        state_next = np.transpose(state_next, (2, 0, 1))
        
        # Add an extra dimension to "reward and "done",
        # so it matches the other tensors dimension
        reward = np.array([reward])
        done   = np.array([done])
        
        self.memory.push(state, action, reward, state_next, done)
        
        
    def getAction(self, state, evaluate = False):
        self.actor.eval()
        
        # Get an action from the actor network of this specific state
        state   = torch.FloatTensor(state).to(self.device)
        state   = torch.permute(state[None, :], (0, 3, 1, 2))
        actions = self.actor(state)
        
        # If we are still training add noise, to help with exploration
        if not evaluate:
            actions += torch.normal(mean = 0.0,
                                    std  = self.noise_std).to(self.device)
            
            # Clip the action values to not exceed the boundaries
            actions = torch.clamp(actions, min = self.min_action_val,
                                           max = self.max_action_val)
        
        self.actor.train()
        
        return actions.cpu().detach().numpy()
        

    def update_target_networks(self, tau = None):
        if tau is None:
            tau = self.tau

        # Extract the weights and biases of the networks
        actor_params         = self.actor.state_dict()
        critic_params        = self.critic.state_dict()
        target_actor_params  = self.target_actor.state_dict()
        target_critic_params = self.target_critic.state_dict()
        
        # Iterate over the mu (actor) network and soft-update the parameters
        for name in actor_params:
            actor_params[name] = tau * actor_params[name].clone() +\
                                 (1-tau) * target_actor_params[name].clone()
                                 
        self.target_actor.load_state_dict(actor_params)
        
        # Iterate over the Q (critic) network and soft-update the parameters
        for name in critic_params:
            critic_params[name] = tau * critic_params[name].clone() +\
                                 (1-tau) * target_critic_params[name].clone()
                                 
        self.target_critic.load_state_dict(critic_params)
        

    def update(self):
        # Get a "batch_size" number of samples from our memory buffer
        state, action, reward, state_next, done = self.memory.sample(self.batch_size)
        
        state      = torch.FloatTensor(state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        state_next = torch.FloatTensor(state_next).to(self.device)
        done       = torch.IntTensor(done).to(self.device)
        
        # Put the network into evaluation
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        
        
        # Q loss
        # Calculate y_i (Q-val/(total future reward) for the target net)
        target_actions    = self.target_actor.forward(state_next)
        target_critic_val = self.target_critic.forward(state_next, target_actions)
        y_i = reward + self.gamma * target_critic_val * (1-done)
        
        # Set the critic network back into training mode
        self.critic.train()
        
        # With y_i, we just need Q_val of our online critic net.
        # We do MSE to find the critic loss
        self.critic_optimizer.zero_grad()
        critic_val  = self.critic.forward(state, action)
        critic_loss = self.critic_criterion(y_i, critic_val)
        
        # Backpropegate the loss through the network
        critic_loss.backward()
        self.critic_optimizer.step()
        

        # Policy loss
        # The policy loss is found by taking the mean of the gradient ascendt
        # of the critic network.
        self.critic.eval()
        self.actor_optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        
        # Backpropegate the loss through the network
        actor_loss.backward()
        self.actor_optimizer.step()
        print(f"after optimizer step: [min: { min(self.actor.fc1.bias.grad):.2E}, max: {max(self.actor.fc1.bias.grad):.2E}, avg: {torch.mean(self.actor.fc1.bias.grad):.2E}]")   
        # print(f"\nAfter optimizer step: actor_loss = {actor_loss.cpu().detach().numpy()}\nBias_grad in fc1:\n {self.actor.fc1.bias.grad}")
        
        # Lastly we update the target networks
        self.update_target_networks()
        
        
    def save_models(self, suffix = ''):
        print('\n# # # # #  Saving checkpoint  # # # # #\n')
        self.critic.save_checkpoints(suffix)
        self.actor.save_checkpoints(suffix)
        self.target_critic.save_checkpoints(suffix)
        self.target_actor.save_checkpoints(suffix)
    
    def load_models(self,  suffix = ''):
        print('\n# # # # #  Loading checkpoint  # # # # #\n')
        self.critic.load_checkpoints(suffix)
        self.actor.load_checkpoints(suffix)
        self.target_critic.load_checkpoints(suffix)
        self.target_actor.load_checkpoints(suffix)
    