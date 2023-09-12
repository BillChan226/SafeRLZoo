import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import sys
sys.path.append("/home/gong112/service_backup/work/RL-InmoovRobot/")
import core
from spinup2.utils.logx import EpochLogger
from spinup2.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup2.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import safety_gym

class SafetyBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.logp_goal_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, logp_goal):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.logp_goal_buf[self.ptr] = logp_goal
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, logp_goal=self.logp_goal_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.adv_safe_buf = np.zeros(size, dtype=np.float32)
        
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rew_safe_buf = np.zeros(size, dtype=np.float32)
        
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ret_safe_buf = np.zeros(size, dtype=np.float32)
        
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.val_safe_buf = np.zeros(size, dtype=np.float32)
        
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # self.logp_safe_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, rew_safe, val_safe):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.rew_safe_buf[self.ptr] = rew_safe
        self.val_buf[self.ptr] = val
        self.val_safe_buf[self.ptr] = val_safe
        self.logp_buf[self.ptr] = logp
        # self.logp_safe_buf[self.ptr] = logp_safe
        self.ptr += 1

    def finish_path(self, last_val=0, last_val_safe=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        rews_safe = np.append(self.rew_safe_buf[path_slice], last_val_safe)
        vals_safe = np.append(self.val_safe_buf[path_slice], last_val_safe)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        deltas_safe = rews_safe[:-1] + self.gamma * vals_safe[1:] - vals_safe[:-1]
        self.adv_safe_buf[path_slice] = core.discount_cumsum(deltas_safe, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.ret_safe_buf[path_slice] = core.discount_cumsum(rews_safe, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, ret_safe=self.ret_safe_buf,
                    adv_safe=self.adv_safe_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    # logger_kwargs[logger_kwargs]

    output_dir = logger_kwargs['output_dir'] 
    logger_kwargs['output_dir'] = output_dir + '/goal'
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    logger_kwargs['output_dir'] = output_dir + '/safe'
    logger_safe = EpochLogger(**logger_kwargs)
    logger_safe.save_config(locals())
    
    logger_kwargs['output_dir'] = output_dir + '/eval'
    logger_test = EpochLogger(**logger_kwargs)
    logger_test.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    print('obs_dim',obs_dim)
    print("act_sim", act_dim)

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    # var_counts_safe = tuple(core.count_vars(module) for module in [ac_safe.pi, ac_safe.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
    # logger.log('\nNumber of parameters in safenet: \t pi: %d, \t v: %d\n'%var_counts_safe)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    buf_safety = SafetyBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    # buf_safe = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    kl_beta = 0.01
    ent_gamma = 0.02
    safe_val_update = False
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # adv_safe, logp_old_safe = data['adv_safe'], data['logp_safe']
        # print("adv_safe: ", adv_safe)
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        
        # Policy safety loss
        # pi_safe, logp_safe = ac.pi_safe(obs, act)
        # # ratio_safe = torch.exp(logp_safe - logp_old_safe)
        # ratio_safe = torch.exp(logp_safe - logp_old)
        # clip_adv_safe = torch.clamp(ratio_safe, 1-clip_ratio, 1+clip_ratio) * adv_safe
        # loss_pi_safe = -(torch.min(ratio_safe * adv_safe, clip_adv_safe)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        
        # approx_kl_safe = (logp_old_safe - logp_safe).mean().item()
        # ent_safe = pi_safe.entropy().mean().item()
        # clipped_safe = ratio_safe.gt(1+clip_ratio) | ratio_safe.lt(1-clip_ratio)
        # clipfrac_safe = torch.as_tensor(clipped_safe, dtype=torch.float32).mean().item()
        # pi_info_safe = dict(kl=approx_kl_safe, ent=ent_safe, cf=clipfrac_safe)

        return loss_pi, pi_info #, loss_pi_safe, pi_info_safe

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        obs_safe, ret_safe = data['obs'], data['ret_safe']
        return ((ac.v(obs) - ret)**2).mean(), ((ac.v_safe(obs_safe) - ret_safe)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    pi_safe_optimizer = Adam(ac.pi_safe.parameters(), lr=pi_lr)
    vf_safe_optimizer = Adam(ac.v_safe.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)
    # logger.setup_pytorch_saver(ac_safe)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        # pi_l_old_safe = pi_l_old_safe.item()
        v_l_old, v_l_old_safe = compute_loss_v(data)
        v_l_old = v_l_old.item()
        v_l_old_safe = v_l_old_safe.item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            # pi_safe_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
 
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()
        
        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            vf_safe_optimizer.zero_grad()
            loss_v, loss_v_safe = compute_loss_v(data)
            loss_v.backward()
            loss_v_safe.backward()
            mpi_avg_grads(ac.v) # average grads across MPI processes
            mpi_avg_grads(ac.v_safe)
            vf_optimizer.step()
            vf_safe_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old), 
                     LossV_safe=v_l_old_safe)
        
    
    def compute_loss_pi_safety(data):
        obs, act, adv_safe, logp_old_safe, logp_old_goal = data['obs'], data['act'], data['adv'], data['logp'], data['logp_goal']

        # # Policy loss
        # pi_safe, logp_safe = ac.pi_safe(obs, act)
        # ratio = torch.exp(logp - logp_old)
        # clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        # loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        
        # Policy safety loss
        pi_safe, logp_safe = ac.pi_safe(obs, act)
        ratio_safe = torch.exp(logp_safe - logp_old_safe)
        clip_adv_safe = torch.clamp(ratio_safe, 1-clip_ratio, 1+clip_ratio) * adv_safe
        # loss_pi_policy = -(torch.min(ratio_safe * adv_safe, clip_adv_safe)).mean()
        loss_pi_policy = (torch.min(ratio_safe * adv_safe, clip_adv_safe)).mean()

        
        # Policy deviation loss
        loss_policy_dev_kl = (logp_old_goal - logp_safe).mean()#.item()
        
        # entropy regularization
        loss_entropy_reg = (torch.exp(logp_safe)*logp_safe).mean()
        
        loss_pi_safe = loss_pi_policy + kl_beta * loss_policy_dev_kl + ent_gamma * loss_entropy_reg
        
        # # Useful extra info
        # approx_kl = (logp_old - logp).mean().item()
        # ent = pi.entropy().mean().item()
        # clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        # clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        # pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        
        approx_kl_safe = (logp_old_safe - logp_safe).mean().item()
        ent_safe = pi_safe.entropy().mean().item()
        clipped_safe = ratio_safe.gt(1+clip_ratio) | ratio_safe.lt(1-clip_ratio)
        clipfrac_safe = torch.as_tensor(clipped_safe, dtype=torch.float32).mean().item()
        pi_info_safe = dict(kl=approx_kl_safe, ent=ent_safe, cf=clipfrac_safe)

        return loss_pi_safe, pi_info_safe, kl_beta * loss_policy_dev_kl, ent_gamma * loss_entropy_reg
    
    def compute_loss_v_safe(data):
        obs, ret = data['obs'], data['ret']
        # obs_safe, ret_safe = data['obs'], data['ret_safe']
        return ((ac.v_safe(obs) - ret)**2).mean()

        
    def update_safety():
        data = buf_safety.get()

        pi_l_old_safe, pi_info_old_safe, kl_loss, ent_loss = compute_loss_pi_safety(data)
        pi_l_old_safe = pi_l_old_safe.item()
        # pi_l_old_safe = pi_l_old_safe.item()
        v_l_old_safe = compute_loss_v_safe(data)
        v_l_old_safe = v_l_old_safe.item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_safe_optimizer.zero_grad()
            # pi_safe_optimizer.zero_grad()
            loss_pi_safe, pi_info_safe, _, _  = compute_loss_pi_safety(data)
            kl_safe = mpi_avg(pi_info_safe['kl'])
            
            if kl_safe > 2 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl_safe.'%i)
                break
            
            # print("is leaf", loss_pi_safe.is_leaf)
            # loss_pi.backward()
            loss_pi_safe.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            # print("loss_pi_safe", loss_pi_safe)
            mpi_avg_grads(ac.pi_safe)
            # pi_optimizer.step()
            pi_safe_optimizer.step()
        
        if safe_val_update:
        # # Value function learning
            for i in range(int(train_v_iters/2)):
                # vf_optimizer.zero_grad()
                vf_safe_optimizer.zero_grad()
                loss_v_safe = compute_loss_v_safe(data)
                # loss_v.backward()
                loss_v_safe.backward()
                # mpi_avg_grads(ac.v) # average grads across MPI processes
                mpi_avg_grads(ac.v_safe)
                # vf_optimizer.step()
                vf_safe_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info_safe['kl'], pi_info_old_safe['ent'], pi_info_safe['cf']
        logger_safe.store(LossPi_safe=pi_l_old_safe, LossKL_safe=kl_loss, LossEnt_safe=ent_loss, LossV_safe=v_l_old_safe,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi_safe=(loss_pi_safe.item() - pi_l_old_safe),
                     )

    # Prepare for interaction with environment
    test_freq = 500
    train_safety_freq = 10
    start_time = time.time()
    o, ep_ret, ep_pen, ep_len = env.reset(), 0, 0, 0
    epochs_safety = 30
    load_policy_epoch = 10
    start_safe_training = 30
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o, ep_ret, ep_len = env.reset(), 0, 0 
        ep_sum = 0
        for t in range(local_steps_per_epoch):
            
            # print("o", o)

            a, v, logp, v_safe = ac.step(torch.as_tensor(o, dtype=torch.float32))
            
            next_o, r, d, infos = env.step(a)
                        
            env.render()
            input("step")
            
            r_goal = r
            r_safe = infos.get('cost', 0)
            # r_safe = 0
            ep_ret += r_goal

            ep_pen += r_safe
            # ep_sum += r_safe

            ep_len += 1

            # save and log
            # store(self, obs, act, rew, val, logp, rew_safe, val_safe, logp_safe)
            buf.store(o, a, r_goal, v, logp, r_safe, v_safe)
            logger.store(VVals=v)
            logger.store(VVals_safe=v_safe)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, v_safe = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                    v_safe = 0
                buf.finish_path(v, v_safe)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    
                    logger.store(EpRet4Goal=ep_ret, EpRet4Safety=ep_pen, EpLen4Goal=ep_len)
                o, ep_ret, ep_pen, ep_len = env.reset(), 0, 0, 0
                
        #print('Score',ep_ret)
        
        ###### Training Safety Policy Net ######
        if epoch > start_safe_training and ((epoch % train_safety_freq == 0) or (epoch == epochs-1)):
            if epoch == load_policy_epoch:
                # Create a new state dictionary containing only the mu_net parameters
                new_state_dict = {
                    k: v for k, v in ac.pi.state_dict().items() if "mu_net" in k
                }

                ac.pi_safe.load_state_dict(new_state_dict, strict=False)
                print("safety network policy loaded !")
            
            for epoch_safety in range(epochs_safety):
                
                o, ep_ret_goal, ep_ret_safety, ep_len_safety= env.reset(), 0, 0, 0
                
                for tr in range(local_steps_per_epoch):

                    a, v_safe, logp_safe, v_goal, logp_goal = ac.step_safety(torch.as_tensor(o, dtype=torch.float32))
                    # print("p_goal", torch.exp(logp_goal))
                    # print("p_safe", torch.exp(logp_safe))
                    next_o, r, d, infos = env.step(a)
                    
                    r_goal = r
                    r_safe = infos.get('cost', 0)
                    
                    ep_ret_goal += r_goal
                    
                    ep_ret_safety += r_safe

                    ep_len_safety += 1

                    buf_safety.store(o, a, r_safe, v_safe, logp_safe, logp_goal)
                    logger_safe.store(VVals_safe=v_safe)
                    
                    # Update obs (critical!)
                    o = next_o

                    timeout = ep_len_safety == max_ep_len
                    terminal = d or timeout
                    epoch_ended = tr==local_steps_per_epoch-1

                    if terminal or epoch_ended:
                        
                        if epoch_ended and not(terminal):
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_len_safety, flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if timeout or epoch_ended:
                            _, v_safe, _, _, _ = ac.step_safety(torch.as_tensor(o, dtype=torch.float32))
                        else:
                            v_safe = 0
                        buf_safety.finish_path(v_safe)
                        if terminal:

                            logger_safe.store(EpRet4Goal=ep_ret_goal, EpRet4Safety=ep_ret_safety, EpLen4Safety=ep_len_safety)

                        o, ep_ret_goal, ep_ret_safety, ep_len_safety = env.reset(), 0, 0, 0
                        
                
                update_safety()
                
                logger_safe.log_tabular('Safety Epoch', epoch_safety)
                logger_safe.log_tabular("EpRet4Goal", with_min_and_max=True)
                logger_safe.log_tabular("EpRet4Safety", with_min_and_max=True)
                logger_safe.log_tabular("EpLen4Safety", with_min_and_max=True)
                logger_safe.log_tabular('VVals_safe', with_min_and_max=True)
                logger_safe.log_tabular('LossPi_safe', average_only=True)
                logger_safe.log_tabular('LossKL_safe', average_only=True)
                logger_safe.log_tabular('LossEnt_safe', average_only=True)
                logger_safe.log_tabular("LossV_safe", average_only=True)
                
                logger_safe.dump_tabular()
                
            # kl_beta = min(0.1+epoch*0.0001, 0.2)
        '''     
        # Test model
    
        if (epoch % test_freq == 0) or (epoch == epochs-1):
            o, ep_ret_test, ep_len_test, ep_pen_test = env.reset(), 0, 0, 0
            for te in range(local_steps_per_epoch): #*8):

                a, v_goal, logp_goal, v_safe, logp_safe = ac.step_test(torch.as_tensor(o, dtype=torch.float32))
                
                next_o, r, d, infos = env.step(a)
                
                r_goal = r
                r_safe = infos.get('cost', 0)
                
                #env.render()
                ep_ret_test += r_goal
                
                ep_pen_test += r_safe

                ep_len_test += 1

                logger_test.store(LogP_Goal=logp_goal)
                logger_test.store(LogP_Safe=logp_safe)
                logger_test.store(VVals_Safe=v_safe)
                logger_test.store(VVals_Goal=v_goal)
                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len_test == max_ep_len
                terminal = d or timeout
                epoch_ended = te==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len_test, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _, v_safe, _ = ac.step_test(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                        v_safe = 0
                    if terminal:

                        logger_test.store(EpRet4Goal_test=ep_ret_test, EpRet4Safety_test=ep_pen_test, EpLen_test=ep_len_test)

                    o, ep_ret_test, ep_pen_test, ep_len_test = env.reset(), 0, 0, 0
                    
            # print("test accuracy: ", ep_test_sum/ep_num)
            logger_test.log_tabular('Epoch', epoch)
            logger_test.log_tabular('EpRet4Goal_test', with_min_and_max=True)
            logger_test.log_tabular('EpRet4Safety_test', with_min_and_max=True)
            logger_test.log_tabular('LogP_Goal', with_min_and_max=True)
            logger_test.log_tabular('LogP_Safe', with_min_and_max=True)
            logger_test.log_tabular('VVals_Goal', with_min_and_max=True)
            logger_test.log_tabular('VVals_Safe', with_min_and_max=True)
            logger_test.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger_test.dump_tabular()
            # logger.log_tabular('EpPenSum_test')
       
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)
            logger_safe.save_state({'env': env}, None)
            logger_test.save_state({'env': env}, None)
        '''
        # Perform PPO update!
        update()
        
        # apply the value network on the whole obs space
        if epoch % 50 == 0:
            # create a matrix of all possible states: a box from -5 to 5, with 0.1 as the increment
            # this matrix should be a numpy array of shape (-1, 2)
            env_space = np.arange(-3, 3, 0.02)
            # env_matrix = []
            # for i in env_space:
            #     for j in env_space:
            #         env_matrix.append([i,j])
            std_matrix = []
            std_safe_matrix = []   
            for i in env_space:
                for j in env_space:
                    std, std_safe = ac.get_std(torch.tensor([i,j], dtype=torch.float32))
                    std_matrix.append(std[0])
                    std_safe_matrix.append(std_safe[0])
            # print("env_matrix", np.shape(env_matrix))
            # pi,logp= ac.pi(torch.tensor(env_matrix, dtype=torch.float32))
            # pi = pi.scale
            # print("std", pi)
            # print("ac.pi(torch.tensor(env_matrix, dtype=torch.float32))", ac.pi(torch.tensor(env_matrix, dtype=torch.float32)))
            # with torch.no_grad():
            #     pi, logp =  ac.pi(torch.tensor(env_matrix, dtype=torch.float32))
            #     v_matrix = pi.scale[:,0].detach().numpy()
            #     pi_safe, logp_safe =  ac.pi_safe(torch.tensor(env_matrix, dtype=torch.float32))
            #     v_safe_matrix = pi_safe.scale[:,0].detach().numpy()
                # v_safe_matrix = ac.pi_safe(torch.tensor(env_matrix, dtype=torch.float32))
            
            # with torch.no_grad():
            #     std_matrix, std_safe_matrix = ac.get_std(torch.tensor(env_matrix, dtype=torch.float32))
    
                
            # print("v_matrix", v_matrix)
            # input("v_matrix")
            # v_matrix = v_matrix.detach().numpy()
            # v_safe_matrix = v_safe_matrix.detach().numpy()
            v_matrix = np.array(std_matrix) #.detach().numpy()
            v_safe_matrix = np.array(std_safe_matrix) #.detach().numpy()
            # print("safe_matrix", v_safe_matrix)
            # break
            # v_matrix
            
            import matplotlib.pyplot as plt
            row = 300
            
            #  save v_matrix.reshape(1000,1000) as a csv
            np.savetxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour4/v_matrix" + str(epoch) + ".csv", v_matrix.reshape(row,row), delimiter=",")
            np.savetxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour4/v_safe_matrix" + str(epoch) + ".csv", v_safe_matrix.reshape(row,row), delimiter=",")
            plt.contourf(env_space, env_space, v_matrix.reshape(row,row), cmap=plt.cm.hot)
            C = plt.contour(env_space, env_space, v_matrix.reshape(row,row), colors='black', linewidths=0.8)
            plt.clabel(C, inline=True, fontsize=12)
            plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour4/goal_"+str(epoch)+".png")
            plt.clf()
            plt.contourf(env_space, env_space, v_safe_matrix.reshape(row,row))
            C = plt.contour(env_space, env_space, v_safe_matrix.reshape(row,row), colors='black', linewidths=0.8)
            plt.clabel(C, inline=True, fontsize=12)
            plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour4/safe_"+str(epoch)+".png")
            plt.clf()
            # break
        
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet4Goal', with_min_and_max=True)
        
        logger.log_tabular('EpRet4Safety', with_min_and_max=True)
        # logger.log_tabular('EpPenSum')
        logger.log_tabular('EpLen4Goal', average_only=True)
        # logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('VVals_safe', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('LossPi_safe', average_only=True)
        logger.log_tabular('LossV_safe', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
    return #ac, env
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='CartPole-v0')
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--env', type=str, default='InmoovGymEnv-v0')
    parser.add_argument('--env', type=str, default='JakaSafeGymEnv-v0')
    #parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--exp_name', type=str, default='PS2_PointSafe')
    parser.add_argument('--output_dir', type=str, default='/home/gong112/service_backup/work/zhaorun/PSGCL/data/')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    from safety_gym.envs.engine import Engine

    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'constrain_hazards': True,
        'constrain_pillars': True,
        'constrain_vases': True,
        'hazards_num': 0,
        'pillars_num': 2,
        'vases_num': 1,
        'randomize_layout': False,
        'continue_goal': False,
        '_seed': 3
    }
    env = Engine(config)
    
    from gym.envs.registration import register

    register(id='SafexpTestEnvironment-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config})

    args.env = "SafexpTestEnvironment-v0"
    
    from spinup2.utils.run_utils import setup_logger_kwargs
    for args.seed in range(0, 10):#0,10
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.output_dir)
        ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
            seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
            logger_kwargs=logger_kwargs)
        
        # def generate_session(model, env, t_max=1000):
            
        #     states, traj_probs, actions, rewards = [], [], [], []
        #     s = env.reset()
        #     q_t = 0   # log probability
            
        #     for t in range(t_max):
        #         # print("state", s)
        #         # input()
        #         # action_probs = self.predict_probs(np.array([s]))[0]
                
        #         a, v, logp = model.step(torch.as_tensor(s, dtype=torch.float32))
                
        #         # a = np.random.choice(self.n_actions,  p = action_probs)
        #         # print("env.step(a)", env.step(a))
        #         new_s, r, done, info = env.step(a)
                
        #         # print("a: ", a)
        #         # print("logp: ", logp)
        #         # input()
        #         # action_prob = 
        #         # q_t *= action_probs[a]
        #         q_t += logp

        #         states.append(s)
        #         traj_probs.append(q_t)
        #         actions.append(a)
        #         rewards.append(r)
                
        #         # print("states: ", states)
        #         # print("actions: ", actions)
        #         # input()

        #         s = new_s
        #         if done:
        #             break
                
        #     print("eval rewards: ", np.sum(rewards))

        #     return states, actions, rewards
    
        # num_expert = 500

        # expert_samples = np.array([generate_session(model, env) for i in range(num_expert)])
        # np.save('expert_samples/pg_cartpole', expert_samples)
