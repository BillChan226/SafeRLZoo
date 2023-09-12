import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)[0]
        mu = self._distribution(obs)[1][0]
        std = self._distribution(obs)[2][0]
        # print("mu", mu)
        # print("std", std)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std), mu, std

    def _log_prob_from_distribution(self, pi, act):
        # print("pi:", pi)
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
class MLPGaussianActor4Safety(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # log_std = 20 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation=nn.Tanh)
        self.log_std_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        # print("self.log_std_net:", self.log_std_net)
        
        # for param in self.mu_net.parameters():
        #     param.data.zero_()
        # self._init_weights(self.mu_net)

    # def _distribution(self, obs):
    #     mu = self.mu_net(obs)
    #     std = torch.exp(self.log_std)
    #     # std = torch.exp(self.log_std_net(obs))
    #     return Normal(mu, std), mu, std
    
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        raw_std = self.log_std_net(obs)  # Unbounded output
        # bounded_std = 3 + 3*raw_std  # Scale and shift to desired range
        # print("bounded_std", bounded_std)
        # std = torch.exp(bounded_std)
        std_min = torch.tensor(0.01)
        mu = torch.clamp(mu,-2,2)
        std = torch.clamp(raw_std + 1,0.01,2)
        
        return Normal(mu, std), mu, std

    def _log_prob_from_distribution(self, pi, act):
        # print("pi:", pi)
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPCritic4Safety(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        
        for param in self.v_net.parameters():
            param.data.zero_()

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # print("lala!")

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
            # self.pi_safe = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
            self.pi_safe = MLPGaussianActor4Safety(obs_dim, action_space.shape[0], hidden_sizes, activation)
            # print("here!")
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
            self.pi_safe = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.v_safe  = MLPCritic4Safety(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi, mu_goal, std_goal = self.pi._distribution(obs)
            pi_safe, mu_safe, std_safe = self.pi_safe._distribution(obs)
        
            # product of two distributions
            # k = std_goal**2 / (std_goal**2+std_safe**2)
            
            # mu_fuse = mu_goal + k*(mu_safe-mu_goal)
            # std_fuse = std_goal**2 - k*std_goal**2
            # std_fuse = torch.sqrt(std_fuse)
            
            # normal_fuse = Normal(mu_fuse, std_fuse)
            
            # a = normal_fuse.sample()
            
            a = pi.sample()
            
            # print("mu_goal:", mu_goal)
            # print("mu_safe:", mu_safe)
            # print("mu_fuse:", mu_fuse)
            # print("std_goal: ", std_goal)
            # print("std_safe: ", std_safe)
            # print("std_fuse: ", std_fuse)
            # print("K: ", k)
            
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            logp_a_safe = self.pi_safe._log_prob_from_distribution(pi_safe, a)
            v = self.v(obs)
            v_safe = self.v_safe(obs)
            
        return a.numpy(), v.numpy(), logp_a.numpy(), v_safe.numpy(), logp_a_safe.numpy()
        # return pi, v
        
        
    def step_test(self, obs):
        with torch.no_grad():
            pi, mu_goal, std_goal = self.pi._distribution(obs)
            pi_safe, mu_safe, std_safe = self.pi_safe._distribution(obs)
        
            # product of two distributions
            k = std_goal**2 / (std_goal**2+std_safe**2)
            
            mu_fuse = mu_goal + k*(mu_safe-mu_goal)
            std_fuse = std_goal**2 - k*std_goal**2
            std_fuse = torch.sqrt(std_fuse)
            
            normal_fuse = Normal(mu_fuse, std_fuse)
            
            a = normal_fuse.sample()
            
            # a = pi.sample()
            
            # print("mu_goal:", mu_goal)
            # print("mu_safe:", mu_safe)
            # print("mu_fuse:", mu_fuse)
            # print("std_goal: ", std_goal)
            # print("std_safe: ", std_safe)
            # print("std_fuse: ", std_fuse)
            # print("K: ", k)
            
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            logp_a_safe = self.pi_safe._log_prob_from_distribution(pi_safe, a)
            v = self.v(obs)
            v_safe = self.v_safe(obs)
            
        return a.numpy(), v.numpy(), logp_a.numpy(), v_safe.numpy(), logp_a_safe.numpy()
        # return pi, v

    def act(self, obs):
        return self.step(obs)[0]