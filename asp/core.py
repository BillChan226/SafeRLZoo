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
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation) #, output_activation=nn.Tanh)
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
        bounded_std = raw_std  # Scale and shift to desired range
        # print("bounded_std", bounded_std)
        std = torch.exp(bounded_std)
        # std_min = torch.tensor(0.01)
        # mu = torch.clamp(mu,-2,2)
        std = torch.clamp(std,-1000,1000)
        # print("mu", mu)
        # print("std", std)
        mu = torch.clamp(mu, -10, 10) + 1
        
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
        
        # for param in self.v_net.parameters():
        #     param.data.zero_()

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(256,256), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
            # self.pi_safe = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
            self.pi_safe = MLPGaussianActor4Safety(obs_dim, action_space.shape[0], hidden_sizes, activation)

        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
            self.pi_safe = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.v_safe  = MLPCritic4Safety(obs_dim, hidden_sizes, activation)
        self.adv_safe  = MLPCritic4Safety(obs_dim+action_space.shape[0], hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi, mu_goal, std_goal = self.pi._distribution(obs)
            # pi_safe, mu_safe, std_safe = self.pi_safe._distribution(obs)
        
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
            # logp_a_safe = self.pi_safe._log_prob_from_distribution(pi_safe, a)
            v = self.v(obs)
            v_safe = self.v_safe(obs)
            
        return a.numpy(), v.numpy(), logp_a.numpy(), v_safe.numpy()#, logp_a_safe.numpy()
    
    def get_std(self, obs):
        with torch.no_grad():
            pi, mu_goal, std_goal = self.pi._distribution(obs)
            pi_safe, mu_safe, std_safe = self.pi_safe._distribution(obs)
        return std_goal.numpy(), std_safe.numpy()
        

    def step_safety(self, obs):
        with torch.no_grad():

            pi_safe, mu_safe, std_safe = self.pi_safe._distribution(obs)
            
            pi_goal, mu_goal, std_goal = self.pi._distribution(obs)
        
            a = pi_safe.sample()
            
            logp_a_safe = self.pi_safe._log_prob_from_distribution(pi_safe, a)
            logp_a_goal = self.pi._log_prob_from_distribution(pi_goal, a)
            
            v_safe = self.v_safe(obs)
            
            v = self.v(obs)
            
        return a.numpy(), v_safe.numpy(), logp_a_safe.numpy(), v.numpy(), logp_a_goal.numpy()
        
        
    def step_test(self, obs):
        with torch.no_grad():
            pi, mu_goal, std_goal = self.pi._distribution(obs)
            pi_safe, mu_safe, std_safe = self.pi_safe._distribution(obs)
        
            # product of two distributions
            k = std_goal**2 / (std_goal**2+std_safe**2)
            
            mu_fuse = mu_goal + k*(mu_safe-mu_goal)
            std_fuse = std_goal**2 - k*std_goal**2
            std_fuse = torch.sqrt(std_fuse) 
            pi_fuse = Normal(mu_fuse, std_fuse) 
            a = pi_fuse.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            logp_a_safe = self.pi_safe._log_prob_from_distribution(pi_safe, a)
            v = self.v(obs)
            v_safe = self.v_safe(obs)
            
        return a.numpy(), v.numpy(), logp_a.numpy(), v_safe.numpy(), logp_a_safe.numpy()
    
    def step_test2(self, obs):
        with torch.no_grad():
            pi, mu_goal, std_goal = self.pi._distribution(obs)
            pi_safe, mu_safe, std_safe = self.pi_safe._distribution(obs)
            a = None
            # # product of two distributions
            # k = std_goal**2 / (std_goal**2+std_safe**2)
            
            # mu_fuse = mu_goal + k*(mu_safe-mu_goal)
            # std_fuse = std_goal**2 - k*std_goal**2
            # std_fuse = torch.sqrt(std_fuse) 
            # pi_fuse = Normal(mu_fuse, std_fuse) 
            sample_set = 10
            action_set = []
            
            logp_safe_max = -99999
            max_a_safe = -1
            for s in range(sample_set):
                action = pi.sample()
                
                logp_a_sample = self.pi._log_prob_from_distribution(pi, action)
                logp_a_safe_sample = self.pi_safe._log_prob_from_distribution(pi_safe, action)
                if logp_a_safe_sample > logp_safe_max:
                    a = action
                    logp_a = logp_a_sample
                    logp_a_safe = logp_a_safe_sample
                    max_a_safe = s
                    logp_safe_max = logp_a_safe_sample
            
            # if a == None:
            #     a = pi.sample()
            # print("max_a_safe", max_a_safe)
                
            v = self.v(obs)
            v_safe = self.v_safe(obs)
            
        return a.numpy(), v.numpy(), logp_a.numpy(), v_safe.numpy(), logp_a_safe.numpy()


    def act(self, obs):
        return self.step(obs)[0]