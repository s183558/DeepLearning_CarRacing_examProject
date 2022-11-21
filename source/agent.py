import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import Q_network, mu_network
from utils import Memory

class DDPGAgent:
    def __init__(self,
                 lr_P,
                 lr_Q,
                 gamma,
                 tau):

        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = 256
        
        # Randomly initialize the critic (Q) and actor (mu) network
        self.Q, self.QT = Q_network(), Q_network(),
        self.P, self.PT = mu_network(), mu_network()

        self.QT.load_state_dict(self.Q.state_dict())
        self.PT.load_state_dict(self.P.state_dict())

        self.memory = Memory(self.max_memory_size)

        self.Q_criterion = nn.MSELoss()
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr_Q)
        # self.P_optimizer = optim.Adam(self.P.parameters(), lr=lr_P)
        self.P_optimizer = optim.Adam(self.P.parameters(), lr=0.01)


    def getAction(self, state):
        action = self.P.forward(state)
        return action.detach().numpy()

    def update(self, batch_size):
        state, action, reward, state_next, done = self.memory.sample(batch_size)
        self.P.zero_grad()
        self.Q.zero_grad()
        self.P_optimizer.zero_grad()

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        state_next = torch.FloatTensor(state_next)
        done = torch.BoolTensor(done)

        # Q loss
        #Qvals = self.Q.forward(state, action)
        #action_next = self.PT.forward(state_next)
        #next_Q = self.QT.forward(state_next, action_next.detach())
        #Qprime = reward + self.gamma * next_Q * (1-done)
        #Q_loss = self.Q_criterion(Qvals, Qprime)

        # Policy loss
        P_loss = -self.Q.forward(state, self.P.forward(state)).mean()/1000
        # P_loss = -(self.Q.forward(state, self.P.forward(state))).mean()

        # P_loss = (self.P.forward(state) ** 2 ).sum()
        # update networks
        P_loss.backward()

        #print(f'Parameters of P: {self.Q.parameters}')
        #print(f'P_loss weifths: {P_loss.weights}')
        #print(f'P_loss gradient: {P_loss.grad}')
        print(f"\n.-.-.-.--.-.-.-.-\n\nbf optimizer step: \n{ self.P.fc1.bias.grad}")        
        self.P_optimizer.step()
        print(f"\nAfter optimizer step: P_loss = {P_loss.detach().numpy()}\nBias_grad in fc1:\n {self.P.fc1.bias.grad}")
        return 
        # self.Q_optimizer.zero_grad()
        # Q_loss.backward()
        # self.Q_optimizer.step()
        return
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