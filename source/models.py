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
                 conv1_dim    = 2,
                 conv2_dim    = 4,
                 cnn_kernel   = 5,
                 pool_kernel  = 2,
                 fc1_dims     = 400,
                 fc2_dims     = 300):

        super(CriticNetwork, self).__init__()
        
        # Filepath to save the parameters of the model
        self.chkpt_dir = chkpt_dir
        self.name = name
        
        # The convolutional layers
        self.conv1 = nn.Conv2d(input_size, conv1_dim, cnn_kernel)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.conv2 = nn.Conv2d(conv1_dim, conv2_dim, cnn_kernel)

        # Calculating the output size of the CNN-network
        state_size = np.asarray(state_size)
        lin_input_size = self._cnn_size_check(state_size, cnn_kernel, pool_kernel)
        lin_input_size = self._cnn_size_check(lin_input_size, cnn_kernel, pool_kernel)

        # Linear network
        # The weights and bias' are initiated with the Xavier uniform distribution
        self.fc1 = nn.Linear((lin_input_size.prod() * conv2_dim) + 3, fc1_dims)
        f1 = 1/ np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1/ np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)
        
        self.q   = nn.Linear(fc2_dims, 1)
        f3 = 0.003 #The paper recomended this
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)
        

    def forward(self, state, action):
        # The first and 2nd convo + pooling layer
        state = self.pool(F.relu(self.conv1(state)))
        state = self.pool(F.relu(self.conv2(state)))

        # Flatten the matrix to a vector, to be used in a fully-connected layer
        state_action_value = torch.cat([torch.flatten(state, 1), action], 1)
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
            
    
    def save_checkpoints(self, suffix = ''):
        checkpoint_file = os.path.join(self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        torch.save(self.state_dict(), checkpoint_file)
        
    def load_checkpoints(self, suffix = ''):
        checkpoint_file = os.path.join(self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        self.load_state_dict(torch.load(checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, name = 'actor', chkpt_dir = 'tmp_models',
                 state_size   = (96, 96),
                 input_size   = 1,
                 conv1_dim    = 2,
                 conv2_dim    = 4,
                 cnn_kernel   = 5,
                 pool_kernel  = 2,
                 fc1_dims     = 400,
                 fc2_dims     = 300,
                 n_actions    = 3):

        super(ActorNetwork, self).__init__()
        
        # Filepath to save the parameters of the model
        self.chkpt_dir = chkpt_dir
        self.name = name
        
        # The convolutional layers
        self.conv1 = nn.Conv2d(input_size, conv1_dim, cnn_kernel)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.conv2 = nn.Conv2d(conv1_dim, conv2_dim, cnn_kernel)

        # Calculating the output size of the CNN-network
        state_size = np.asarray(state_size)
        lin_input_size = self._cnn_size_check(state_size, cnn_kernel, pool_kernel)
        lin_input_size = self._cnn_size_check(lin_input_size, cnn_kernel, pool_kernel)

        # Linear network
        # The weights and bias' are initiated with the Xavier uniform distribution
        self.fc1 = nn.Linear(lin_input_size.prod() * conv2_dim, fc1_dims)
        f1 = 1/ np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1/ np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)
        
        self.mu  = nn.Linear(fc2_dims, n_actions)
        f3 = 0.003 #The paper recomended this
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        
    def forward(self, state):
        # The first and 2nd convo + pooling layer
        state = self.pool(F.relu(self.conv1(state)))
        state = self.pool(F.relu(self.conv2(state)))
        
        # Flatten the matrix to a vector, to be used in a fully-connected layer
        prob = torch.flatten(state, 1)
        prob = F.relu(self.bn1(self.fc1(prob)))
        prob = F.relu(self.bn2(self.fc2(prob)))
        
        # Run tanh on the steering (-1, 1) and sigmoid on gas and breaking (0,1)
        mu = self.mu(prob)
        mu[:, 0] = torch.tanh(mu[:, 0])
        mu[:, 1:] = torch.sigmoid(mu[:, 1:])
        
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
    
    
    def save_checkpoints(self, suffix = ''):
        checkpoint_file = os.path.join(self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        torch.save(self.state_dict(), checkpoint_file)
        
        
    def load_checkpoints(self, suffix = ''):
        checkpoint_file = os.path.join(self.chkpt_dir,
                                       self.name + "_ddpg" + suffix)
        self.load_state_dict(torch.load(checkpoint_file))
        
        
        