import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import Q, P
from utils import Memory


class DDPGAgent:
    def __init__(
        self,
        lr_P,
        lr_Q,
        gamma,
        tau,
    ):

        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = 256

        self.Q, self.QT, self.P, self.PT = Q(), Q(), P(), P()

        self.QT.load_state_dict(self.Q.state_dict())
        self.PT.load_state_dict(self.P.state_dict())

        self.memory = Memory(self.max_memory_size)

        self.Q_criterion = nn.MSELoss()
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr_Q)
        self.P_optimizer = optim.Adam(self.P.parameters(), lr=lr_P)

    def getAction(self, state):
        action = self.P.forward(state)
        return action.detach().numpy()

    def update(self, batch_size):

        state, action, reward, state_next, _ = self.memory.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        state_next = torch.FloatTensor(state_next)

        # Q loss
        Qvals = self.Q.forward(state, action)
        action_next = self.PT.forward(state_next)
        next_Q = self.QT.forward(state_next, action_next.detach())
        Qprime = reward + self.gamma * next_Q
        Q_loss = self.Q_criterion(Qvals, Qprime)

        # Policy loss
        P_loss = -self.Q.forward(state, self.P.forward(state)).mean()

        # update networks
        self.P_optimizer.zero_grad()
        P_loss.backward()
        self.P_optimizer.step()

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # update target networks
        for parameters, parameters_target in zip(
            self.P.parameters(), self.PT.parameters()
        ):
            parameters_target.data.copy_(
                parameters.data * self.tau + parameters_target.data * (1.0 - self.tau)
            )

        for parameters, parameters_target in zip(
            self.Q.parameters(), self.QT.parameters()
        ):
            parameters_target.data.copy_(
                parameters.data * self.tau + parameters_target.data * (1.0 - self.tau)
            )
