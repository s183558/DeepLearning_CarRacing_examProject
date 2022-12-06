import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


# Value network (Q-network)
class CriticNetwork(nn.Module):
    def __init__(self, name = 'critic', chkpt_dir = 'tmp_models',
                 state_size   = (96, 96),
                 input_size   = 1,
                 conv1_dim    = 6,
                 conv2_dim    = 12,
                 cnn_kernel1  = 7,
                 cnn_kernel2  = 4,
                 pool_kernel  = 2,
                 fc1_dims     = 216,
                 fc2_dims     = 300):

        super(CriticNetwork, self).__init__()
        
        # Filepath to save the parameters of the model
        self.chkpt_dir = chkpt_dir
        self.name = name
        
        # The convolutional layers
        self.conv1 = nn.Conv2d(input_size, 16, kernel_size = 5, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, stride=3)

        # Linear network
        # The weights and bias' are initiated with the Xavier uniform distribution
        self.fc1 = nn.Linear(129, 64)
        self.fc2 = nn.Linear(64, 32)
        self.q = nn.Linear(32, 1)
        
    def forward(self, state, action):
        # The first and 2nd convo + pooling layer
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        # Flatten the matrix to a vector, to be used in a fully-connected layer
        state_action_value = torch.cat([torch.flatten(state, 1), action[:,:1]], 1)
        state_action_value = F.relu(self.fc1(state_action_value))
        state_action_value = F.relu(self.fc2(state_action_value))
        q = self.q(state_action_value)
        
        return q

    def _cnn_size_check(self, img_size, cnn_kernel_size, pool_kernel_size):
        # If the image size is divisible by the pooling kernel, and will be
        # atleast 1 pixel big after the whole convolution process.
        if (img_size % pool_kernel_size == 0).all() and \
           (img_size > ((cnn_kernel_size-1) * pool_kernel_size)).all():
            
            # The image size after 1 full convolutional layer.
            lin_input_size = (img_size - (cnn_kernel_size-1)) / pool_kernel_size
            lin_input_size = lin_input_size.astype(int)
            return lin_input_size
        else:
            print(f"ERROR\nThe size for the linear input is wrong: {img_size}")
            
    
    def save_checkpoints(self, suffix = '', drive_dir = ''):
        checkpoint_file = os.path.join(drive_dir + self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        torch.save(self.state_dict(), checkpoint_file)
        
    def load_checkpoints(self, suffix = '', drive_dir = ''):
        checkpoint_file = os.path.join(drive_dir + self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        self.load_state_dict(torch.load(checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, name = 'actor', chkpt_dir = 'tmp_models',
                 state_size   = (96, 96),
                 input_size   = 1,
                 conv1_dim    = 6,
                 conv2_dim    = 12,
                 cnn_kernel1  = 7,
                 cnn_kernel2  = 4,
                 pool_kernel  = 2,
                 fc1_dims     = 216,
                 fc2_dims     = 300,
                 n_actions    = 2):

        super(ActorNetwork, self).__init__()
        
        # Filepath to save the parameters of the model
        self.chkpt_dir = chkpt_dir
        self.name = name
        
        # The convolutional layers
        self.conv1 = nn.Conv2d(input_size, 16, kernel_size = 5, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, stride=3)

        # Linear network
        # The weights and bias' are initiated with the Xavier uniform distribution
        self.fc1 = nn.Linear(128, 64)
        f = 0.005
        torch.nn.init.uniform_(self.fc1.weight.data, -f, f)
        torch.nn.init.uniform_(self.fc1.bias.data, -f, f)
        
        
        self.mu = nn.Linear(64, n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f, f)
        torch.nn.init.uniform_(self.mu.bias.data, -f, f)
        
        
    def forward(self, state):
        # The first and 2nd convo + pooling layer
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        
        # Flatten the matrix to a vector, to be used in a fully-connected layer
        prob = torch.flatten(state, 1)
        prob = F.relu(self.fc1(prob))
        
        # Run tanh on the action (-1, 1) split the acceleration into gas and breaking
        mu = torch.tanh(self.mu(prob))/4

        mu = torch.cat([mu, torch.relu(-mu[:, 1:])],1)
        mu[:, 1:] = torch.relu(mu[:, 1:])
        
        # Make the steering a fourth, as the steering is really hard
        # u[:,0] /= 4
        
        return mu

    def _cnn_size_check(self, img_size, cnn_kernel_size, pool_kernel_size):
        # If the image size is divisible by the pooling kernel, and will be
        # atleast 1 pixel big after the whole convolution process.
        if (img_size % pool_kernel_size == 0).all() and \
           (img_size > ((cnn_kernel_size-1) * pool_kernel_size)).all():
            
            # The image size after 1 full convolutional layer.
            lin_input_size = (img_size - (cnn_kernel_size-1)) / pool_kernel_size
            lin_input_size = lin_input_size.astype(int)
            return lin_input_size
        else:
            print(f"ERROR\nThe size for the linear input is wrong: {img_size}")
    
    
    def save_checkpoints(self, suffix = '', drive_dir = ''):
        checkpoint_file = os.path.join(drive_dir + self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        torch.save(self.state_dict(), checkpoint_file)
        
        
    def load_checkpoints(self, suffix = '', drive_dir = ''):
        checkpoint_file = os.path.join(drive_dir + self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        self.load_state_dict(torch.load(checkpoint_file))
        
        