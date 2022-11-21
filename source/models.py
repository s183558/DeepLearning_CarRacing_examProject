import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Q_network(nn.Module):
    def __init__(self,
                 state_size   = (96, 96),
                 input_size   = 1,
                 cnn_l1_size  = 2,
                 cnn_l2_size  = 4,
                 lin_l1_size  = 100,
                 lin_out_size = 1,
                 batch_size   = 32,
                 cnn_kernel   = 5,
                 pool_kernel  = 2):

        super().__init__()

        self.conv1 = nn.Conv2d(input_size, cnn_l1_size, cnn_kernel)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.conv2 = nn.Conv2d(cnn_l1_size, cnn_l2_size, cnn_kernel)

        # Calculating the output size of the CNN-network
        state_size = np.asarray(state_size)
        lin_input_size = self._cnn_size_check(state_size, cnn_kernel, pool_kernel)
        lin_input_size = self._cnn_size_check(lin_input_size, cnn_kernel, pool_kernel)

        # Linear network
        self.fc1 = nn.Linear((lin_input_size.prod() * cnn_l2_size) + 3, lin_l1_size)
        self.fc2 = nn.Linear(lin_l1_size, lin_out_size)

    def forward(self, state, action):
        # Turn the image into a tensor if isn't already one
        if type(state) == np.ndarray:
             # Rearange the dimensions of the tensor to fit PyTorch
            if len(state.shape) == 2:
                state = torch.from_numpy(np.atleast_3d(state)).float()
                state = torch.permute(state, (2, 0, 1))
            else:
                state = torch.from_numpy(np.atleast_3d(state)).float()


        # add the one dimension of color (grayscale)
        state = state[None, :] 
        state = torch.permute(state, (1, 0, 2, 3)) #(batch, channels, w, h)
        #print(f'Q forward state shape: {state.shape}')
        

        # The first and 2nd convo + pooling layer
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the matrix to a vector, to be used in a fully-connected layer
        x = torch.cat([torch.flatten(x, 1), action], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

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
            
        


class mu_network(nn.Module):
    def __init__(self,
                 state_size   = (96, 96),
                 input_size   = 1,
                 cnn_l1_size  = 2,
                 cnn_l2_size  = 4,
                 lin_l1_size  = 100,
                 lin_out_size = 3,
                 batch_size   = 32,
                 cnn_kernel   = 5,
                 pool_kernel  = 2):

        super().__init__()

        self.conv1 = nn.Conv2d(input_size, cnn_l1_size, cnn_kernel)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.conv2 = nn.Conv2d(cnn_l1_size, cnn_l2_size, cnn_kernel)

        # Calculating the output size of the CNN-network
        state_size = np.asarray(state_size)
        lin_input_size = self._cnn_size_check(state_size, cnn_kernel, pool_kernel)
        lin_input_size = self._cnn_size_check(lin_input_size, cnn_kernel, pool_kernel)

        # Linear network
        self.fc1 = nn.Linear(lin_input_size.prod() * cnn_l2_size, lin_l1_size)
        self.fc2 = nn.Linear(lin_l1_size, lin_out_size)
    
    def forward(self, state):
        # Turn the image into a tensor if isn't already one
        if type(state) == np.ndarray:
             # Rearange the dimensions of the tensor to fit PyTorch
            if len(state.shape) == 2:
                state = torch.from_numpy(np.atleast_3d(state)).float()
                state = torch.permute(state, (2, 0, 1))
            else:
                state = torch.from_numpy(np.atleast_3d(state)).float()
        

        # add the one dimension of color (grayscale)
        state = state[None, :] 
        state = torch.permute(state, (1, 0, 2, 3)) #(batch, channels, w, h)
        #print(f'Mu forward state shape: {state.shape}')
 
        # The first and 2nd convo + pooling layer
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the matrix to a vector, to be used in a fully-connected layer
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x[:, 0] = torch.tanh(x[:, 0])
        x[:, 1:] = torch.sigmoid(x[:, 1:])
       
       
        #print(f'gas before we clip it: {x[:, 1]}')
        #x[:, 1] = torch.clamp(x[:, 1], min=0.1, max=None)

        #print(f'mu forward, output valus: {x}')
        
        return x

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
