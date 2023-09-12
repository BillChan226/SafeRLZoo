import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from experts.PG import PG
from torch.optim.lr_scheduler import StepLR
import math
import matplotlib
from collections import namedtuple
import collections
from itertools import count
import timeit
from datetime import timedelta
from PIL import Image
from tensorboardX import SummaryWriter
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p

import sys
sys.path.append("/home/gong112/service_backup/work/RL-InmoovRobot/")
from environments.registry import *

# env = KukaDiverseObjectEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)
# env.cid = p.connect(p.DIRECT)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# env_name = 'InmoovGymEnv-v0'
env_name = 'KukaButtonGymEnv-v0'
env = gym.make(env_name).unwrapped
if seed is not None:
    env.seed(seed)
state = env.reset()

n_actions = env.action_space.n
state_shape = env.observation_space.shape

print("n_actions", n_actions)
print("state_shape", state_shape)

# initializing a model
model = PG(state_shape, n_actions)

mean_rewards = []
for i in range(100):
    rewards = [model.train_on_env(env) for _ in range(5)] 
    mean_rewards.append(np.mean(rewards))
    print("iteration: ", i)
    print("mean_rewards: ", np.mean(rewards))
    if i % 5:
        print("mean reward:%.3f" % (np.mean(rewards)))
        plt.figure(figsize=[9, 6])
        plt.title("Mean reward per 100 games")
        plt.plot(mean_rewards)
        plt.grid()
        # plt.show()
        plt.savefig('plots/Kuka/PG_learning_curve.png')
        plt.close()
    
    if np.mean(rewards) > 500:
        print("TRAINED!")
        break

torch.save(model, "experts/saved_expert/kuka.model")
#model.load("experts/saved_expert/pg.model")

num_expert = 100

expert_samples = np.array([model.generate_session(env) for i in range(num_expert)])
np.save('expert_samples/pg_cartpole', expert_samples)
