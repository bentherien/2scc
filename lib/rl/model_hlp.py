"""
This file is a modified version of "model.py" from the following repository
https://github.com/pranz24/pytorch-soft-actor-critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from lib.rl.model import DeterministicPolicy, GaussianPolicy

import types

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetworkHLP(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkHLP, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        # print(action.shape)
        # print(state.shape)
        xu = torch.cat([state, action], 1)

        # print(xu.shape)
        # print(self.linear1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicyHLP(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyHLP, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyHLP, self).to(device)




def modifyGaussianPolicyForward(gpPolicy):

    def forward(self, state, adapter_w):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        x = F.linear(x, adapter_w) + x

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    gpPolicy.forward = types.MethodType(forward, gpPolicy)

    return gpPolicy

    





class DeterministicPolicyHLP(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicyHLP, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, hidden_dim)
        # self.mean = nn.Linear(hidden_dim, num_actions)



        # self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        # if action_space is None:
        #     self.action_scale = 1.
        #     self.action_bias = 0.
        # else:
        #     self.action_scale = torch.FloatTensor(
        #         (action_space.high - action_space.low) / 2.)
        #     self.action_bias = torch.FloatTensor(
        #         (action_space.high + action_space.low) / 2.)

    def forward(self, state, LLPolicy):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        adapter_w = F.relu(self.mean(x)) #* self.action_scale + self.action_bias

        # print(adapter_w.shape)
        # if type(LLPolicy) == DeterministicPolicy:
        #     adapter_w = adapter_w.view(adapter_w.size(0),LLPolicy.linear2.out_features,LLPolicy.mean.in_features)
        # else:
        #     adapter_w = adapter_w.view(adapter_w.size(0),LLPolicy.linear2.out_features,LLPolicy.mean_linear.in_features)

        # print(adapter_w.shape)
        
        return LLPolicy(state, adapter_w)

    def sample(self, state, LLPolicy):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        adapter_w = torch.tanh(self.mean(x)) #* self.action_scale + self.action_bias
        # print(adapter_w.shape)

        # if type(LLPolicy) == DeterministicPolicy:
        #     adapter_w = adapter_w.view(adapter_w.size(0),LLPolicy.linear2.out_features,LLPolicy.mean.in_features)
        # else:
        #     adapter_w = adapter_w.view(adapter_w.size(0),LLPolicy.linear2.out_features,LLPolicy.mean_linear.in_features)

        # print(adapter_w.shape)

        return LLPolicy.sample(state,adapter_w)


        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        # self.action_scale = self.action_scale.to(device)
        # self.action_bias = self.action_bias.to(device)
        # self.noise = self.noise.to(device)
        return super(DeterministicPolicyHLP, self).to(device)
