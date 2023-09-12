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
        self.logp_safe_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, rew_safe, val_safe, logp_safe):
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
        self.logp_safe_buf[self.ptr] = logp_safe
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
                    adv_safe=self.adv_safe_buf, logp_safe=self.logp_safe_buf)
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
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    # print('obs_dim',obs_dim)

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    
    # ac_safe = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

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
    # buf_safe = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, adv_safe, logp_old_safe = data['obs'], data['act'], data['adv'], data['logp'], data['adv_safe'], data['logp_safe']
        # print("adv_safe: ", adv_safe)
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        
        # Policy safety loss
        pi_safe, logp_safe = ac.pi_safe(obs, act)
        # ratio_safe = torch.exp(logp_safe - logp_old_safe)
        ratio_safe = torch.exp(logp_safe - logp_old)
        clip_adv_safe = torch.clamp(ratio_safe, 1-clip_ratio, 1+clip_ratio) * adv_safe
        loss_pi_safe = -(torch.min(ratio_safe * adv_safe, clip_adv_safe)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        
        approx_kl_safe = (logp_old_safe - logp_safe).mean().item()
        ent_safe = pi_safe.entropy().mean().item()
        clipped_safe = ratio_safe.gt(1+clip_ratio) | ratio_safe.lt(1-clip_ratio)
        clipfrac_safe = torch.as_tensor(clipped_safe, dtype=torch.float32).mean().item()
        pi_info_safe = dict(kl=approx_kl_safe, ent=ent_safe, cf=clipfrac_safe)

        return loss_pi, pi_info, loss_pi_safe, pi_info_safe

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

        pi_l_old, pi_info_old, pi_l_old_safe, pi_info_old_safe = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        pi_l_old_safe = pi_l_old_safe.item()
        v_l_old, v_l_old_safe = compute_loss_v(data)
        v_l_old = v_l_old.item()
        v_l_old_safe = v_l_old_safe.item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            pi_safe_optimizer.zero_grad()
            loss_pi, pi_info, loss_pi_safe, pi_info_safe = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            #kl_safe = mpi_avg(pi_info_safe['kl'])
            #if kl_safe > 5 * target_kl:
            #    logger.log('Early stopping at step %d due to reaching max kl_safe.'%i)
            #    break
            loss_pi.backward()
            loss_pi_safe.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            mpi_avg_grads(ac.pi_safe)
            pi_optimizer.step()
            pi_safe_optimizer.step()
        
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
                     LossPi_safe=pi_l_old_safe, LossV_safe=v_l_old_safe)

    # Prepare for interaction with environment
    test_freq = 10
    start_time = time.time()
    o, ep_ret, ep_pen, ep_len = env.reset(), 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o, ep_ret, ep_len = env.reset(), 0, 0 
        ep_sum = 0
        for t in range(local_steps_per_epoch):

            a, v, logp, v_safe, logp_safe = ac.step(torch.as_tensor(o, dtype=torch.float32))
            
            next_o, r, d, infos = env.step(a)
            
            r_goal = r[0]
            r_safe = r[1]
            #env.render()
            ep_ret += r_goal
            # if r_safe == -1:
            if d == True:
                if infos["contact"]==0:
                    r_safe = 10

            ep_pen += r_safe
            # ep_sum += r_safe

            ep_len += 1

            # save and log
            # store(self, obs, act, rew, val, logp, rew_safe, val_safe, logp_safe)
            buf.store(o, a, r_goal, v, logp, r_safe, v_safe, logp_safe)
            logger.store(VVals=v)
            
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
                    _, v, _, v_safe, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                    v_safe = 0
                buf.finish_path(v, v_safe)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    
                    logger.store(EpRet=ep_ret, EpPen=ep_pen, EpLen=ep_len)
                o, ep_ret, ep_pen, ep_len = env.reset(), 0, 0, 0
        #print('Score',ep_ret)
        
        # logger.store(EpPenSum=ep_sum)
        # print("ep_sum", ep_sum)
        # Test model
        ep_test_sum = 0
        ep_num = 0
        if (epoch % test_freq == 0) or (epoch == epochs-1):
            o, ep_ret_test, ep_len_test, ep_pen_test = env.reset(), 0, 0, 0
            for te in range(local_steps_per_epoch):

                a, v, logp, v_safe, logp_safe = ac.step_test(torch.as_tensor(o, dtype=torch.float32))
                
                next_o, r, d, infos = env.step(a)
                
                # if infos["contact"] != 0:
                #     print("infos: ", infos)
                    
                r_goal = r[0]
                r_safe = r[1]
                
                if d == True:
                    if infos["contact"]==0:
                        r_safe = 10
                #env.render()
                ep_ret_test += r_goal
                
                ep_pen_test += r_safe

                ep_len_test += 1

                # Update obs (critical!)
                o = next_o

                timeout = ep_len_test == max_ep_len
                terminal = d or timeout
                epoch_ended = te==local_steps_per_epoch-1
                #print('terminal',terminal)
                #print('done',d)
                #print('ep_len',ep_len)
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
                        ep_test_sum += ep_pen_test
                        logger.store(EpRet_test=ep_ret_test, EpPen_test=ep_pen_test, EpLen_test=ep_len_test)

                    o, ep_ret_test, ep_pen_test, ep_len_test = env.reset(), 0, 0, 0
            # print("ep_test_sum", ep_test_sum)
            # logger.store(EpPenSum_test=ep_test_sum)
                    
            # print("test accuracy: ", ep_test_sum/ep_num)
            logger.log_tabular('EpRet_test', with_min_and_max=True)
            logger.log_tabular('EpPen_test', with_min_and_max=True)
            # logger.log_tabular('EpPenSum_test')
            
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        
        logger.log_tabular('EpPen', with_min_and_max=True)
        # logger.log_tabular('EpPenSum')
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossPi_safe', average_only=True)
        logger.log_tabular('LossV_safe', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
    return ac, env
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='CartPole-v0')
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--env', type=str, default='InmoovGymEnv-v0')
    parser.add_argument('--env', type=str, default='JakaRLGymEnv-v0')
    #parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--steps', type=int, default=80000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='ppo_jaka_pcT_disT_20230616')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup2.utils.run_utils import setup_logger_kwargs
    for args.seed in range(0, 1):#0,10
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
        model, env = ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
            seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
            logger_kwargs=logger_kwargs)
        
        def generate_session(model, env, t_max=1000):
            
            states, traj_probs, actions, rewards = [], [], [], []
            s = env.reset()
            q_t = 0   # log probability
            
            for t in range(t_max):
                # print("state", s)
                # input()
                # action_probs = self.predict_probs(np.array([s]))[0]
                
                a, v, logp = model.step(torch.as_tensor(s, dtype=torch.float32))
                
                # a = np.random.choice(self.n_actions,  p = action_probs)
                # print("env.step(a)", env.step(a))
                new_s, r, done, info = env.step(a)
                
                # print("a: ", a)
                # print("logp: ", logp)
                # input()
                # action_prob = 
                # q_t *= action_probs[a]
                q_t += logp

                states.append(s)
                traj_probs.append(q_t)
                actions.append(a)
                rewards.append(r)
                
                # print("states: ", states)
                # print("actions: ", actions)
                # input()

                s = new_s
                if done:
                    break
                
            print("eval rewards: ", np.sum(rewards))

            return states, actions, rewards
    
        num_expert = 500

        expert_samples = np.array([generate_session(model, env) for i in range(num_expert)])
        np.save('expert_samples/pg_cartpole', expert_samples)