import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_network(nn.Module):
    def __init__(
        self,
        # state_size,
        input_size=1,
        cnn_l1_size=10,
        cnn_l2_size=20,
        lin_l1_size=100,
        lin_out_size=1,
        batch_size=32,
        cnn_kernel=5,
        pool_kernel=2,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(input_size, cnn_l1_size, cnn_kernel)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.conv2 = nn.Conv2d(cnn_l1_size, cnn_l2_size, cnn_kernel)

        # Calculating the output size of the CNN-network
        lin_input_size = self._cnn_size_check((input_size - cnn_kernel) / pool_kernel)
        lin_input_size = self._cnn_size_check(
            (lin_input_size - cnn_kernel) / pool_kernel
        )

        # Linear network
        self.fc1 = nn.Linear(lin_input_size + 3, lin_l1_size)
        self.fc2 = nn.Linear(lin_l1_size, lin_out_size)

    def forward(self, state, action):
        # The first and 2nd convo + pooling layer
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the matrix to a vector, to be used in a fully-connected layer
        x = torch.cat([torch.flatten(x, 1), action], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _cnn_size_check(lin_input_size):
        if lin_input_size % 2 == 0 and lin_input_size > 1:
            lin_input_size = int(lin_input_size)
            return lin_input_size
        else:
            print(f"The size for the linear input is wrong: {lin_input_size}")
            quit


class mu_network(nn.Module):
    def __init__(
        self,
        # state_size,
        input_size=1,
        cnn_l1_size=10,
        cnn_l2_size=20,
        lin_l1_size=100,
        lin_out_size=1,
        batch_size=32,
        cnn_kernel=5,
        pool_kernel=2,
    ):
        self.conv1 = nn.Conv2d(input_size, cnn_l1_size, cnn_kernel)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.conv2 = nn.Conv2d(cnn_l1_size, cnn_l2_size, cnn_kernel)

        # Calculating the output size of the CNN-network
        lin_input_size = self._cnn_size_check((input_size - cnn_kernel) / pool_kernel)
        lin_input_size = self._cnn_size_check(
            (lin_input_size - cnn_kernel) / pool_kernel
        )

        # Linear network
        self.fc1 = nn.Linear(lin_input_size, lin_l1_size)
        self.fc2 = nn.Linear(lin_l1_size, lin_out_size)

        # Initialize buffer - TODO
        self.buffer = list(batch_size)  # Containing tuple's (s_t, a_t, r_t, s_(t+1))

    def forward(self, state):
        # The first and 2nd convo + pooling layer
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the matrix to a vector, to be used in a fully-connected layer
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

    def _cnn_size_check(lin_input_size):
        if lin_input_size % 2 == 0 and lin_input_size > 1:
            lin_input_size = int(lin_input_size)
            return lin_input_size
        else:
            print(f"The size for the linear input is wrong: {lin_input_size}")
            quit


class Q(nn.Module):
    def __init__(self, input_dims=1, action_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dims,
                out_channels=2,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(
                in_channels=2, out_channels=4, kernel_size=5, stride=1, padding=2
            ),
            nn.MaxPool2d(4, 4),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=24 + action_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_size),
        )

    def forward(self, state, action):
        x = self.conv(state)
        x = torch.cat([x, action])
        x = self.fc(x)
        return x


class P(nn.Module):
    def __init__(self, input_dims=1, action_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dims,
                out_channels=2,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(
                in_channels=2, out_channels=4, kernel_size=5, stride=1, padding=2
            ),
            nn.MaxPool2d(4, 4),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=24, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_size),
        )

    def forward(self, state):
        x = self.conv(state)
        x = self.fc(x)
        return x
