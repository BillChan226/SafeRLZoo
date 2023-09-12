import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import sys, os
sys.path.append("/home/gong112/service_backup/work/RL-InmoovRobot/")
import spinup2.algos.pytorch.ppo.core as core
from spinup2.utils.logx import EpochLogger
from spinup2.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup2.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from cost import CostNN
import pickle


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
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
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

    def recalculate_adv(self, last_val=0, start_ptx=-1, end_ptx=-1):
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
        print("in end_ptx", end_ptx)
        path_slice = slice(start_ptx, end_ptx)
        print("path_slice", path_slice)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        # self.path_start_idx = self.ptr

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
                    adv=self.adv_buf, logp=self.logp_buf, done=self.done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
    def view(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        # self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, done=self.done_buf)
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
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print('obs_dim',obs_dim)
    print("act_dim", act_dim)

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
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
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        
    # CONVERTS TRAJ LIST TO STEP LIST
    def preprocess_traj(traj_list, step_list, is_Demo = False):
        step_list = step_list.tolist()
        for traj in traj_list:
            states = np.array(traj[0])
            if is_Demo:
                probs = np.ones((states.shape[0], 1))
            else:
                probs = np.array(traj[1]).reshape(-1, 1)
            actions = np.array(traj[2]).reshape(-1, 1)
            x = np.concatenate((states, probs, actions), axis=1)
            step_list.extend(x)
            
        return np.array(step_list)

    # Prepare for interaction with environment
    
    # cost function approximator for GCL (w.r.t. state + action)
    print("cost function dimension", obs_dim + act_dim)
    cost_f = CostNN(obs_dim + act_dim)
    
    cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
    
          
    # Get a list of all text files in the directory
    # demo_path_dir = "/home/gong112/service_backup/work/zhaorun/PSGCL/demo/1-1"
    # demo_files = [f for f in os.listdir(demo_path_dir) if f.endswith('.txt')]
    # demo_epochs = []
    # # Iterate over the list of text files and read them one by one
    # for demo_file in demo_files:
    #     demo_path = os.path.join(demo_path_dir, demo_file)
    #     # print("demo_path", demo_path)
        
    #     # Read the data from the text file into a NumPy array
    #     data = np.genfromtxt(demo_path, delimiter=',')[:,:6]
        
    #     demo_epochs.append(data)

    
    mean_rewards = []
    mean_costs = []
    mean_loss_rew = []
    EPISODES_TO_PLAY = 1
    REWARD_FUNCTION_UPDATE = 10
    DEMO_BATCH = 1000
    sample_trajs = []

    D_demo, D_samp = np.array([]), np.array([])
    
    demo_path_dir = '/home/gong112/service_backup/work/zhaorun/PSGCL/ppo/demo_live_jaka.pickle'
    with open(demo_path_dir, "rb") as file:
        demo_data = pickle.load(file)
    # print("demo_data", demo_data["obs"][:10])
    # input()
    demo_data["obs"] = np.array(demo_data["obs"])[:,:6]
    demo_logp = np.zeros((len(demo_data["obs"]), 1))

    expert_trajs = np.concatenate((demo_data["obs"], demo_data["act"], demo_logp.reshape(-1,1)), axis=1)

    print("expert_trajs", np.shape(expert_trajs))
    D_demo = expert_trajs
    
    # D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
    return_list, sum_of_cost_list = [], []
    
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0


    for epoch in range(epochs): # epochs to optimize the cost function and policy
        
        # sampling trajectories w.r.t. current policy
        o, ep_ret, ep_len = env.reset(), 0, 0 
        for t in range(local_steps_per_epoch):

            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            # print("a", a)
            next_o, r, d, _ = env.step(a)
            # r = 0
            #env.render()
            ep_ret += r
            
            r = 0

            ep_len += 1
            if d == False:
                d_i = 0
            if d == True:
                d_i = 1
            # save and log
            buf.store(o, a, r, v, logp, d_i)
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
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    buf.done_buf[buf.ptr-1] = 2 # unsuccessful done
                else:
                    v = 0
                
                buf.finish_path(v)
                if terminal or epoch_ended:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
        #print('Score',ep_ret)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)
            
            
        # dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        # just view the data but update the policy
        trajs_dict = buf.view()
        # trajs = [trajs_dict["obs"].numpy(), trajs_dict["act"].numpy(), trajs_dict["logp"].numpy()]
        # print("obs", np.shape(trajs_dict["obs"].numpy()))
        # print("act", np.shape(trajs_dict["act"].numpy()))
        # print("logp", np.shape(trajs_dict["logp"].numpy().reshape(-1,1)))
        trajs = np.concatenate((trajs_dict["obs"].numpy().tolist(), trajs_dict["act"].numpy().tolist(), trajs_dict["logp"].numpy().reshape(-1,1)), axis=1).tolist()
        # print("trajs", trajs)
        # input()
        # print("shape", np.shape(trajs))

        sample_trajs.extend(trajs)
        D_samp = np.array(sample_trajs)

        # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
        loss_rew = []
        loss_ioc_episode = []
        for _ in range(REWARD_FUNCTION_UPDATE):
            
            selected_samp = np.random.choice(len(D_samp), DEMO_BATCH).tolist()
            selected_demo = np.random.choice(len(D_demo), DEMO_BATCH).tolist()
            
            # print("selected_samp", selected_samp)
            # print("selected_demo", selected_demo)
            # print("D_samp", D_samp[:10])
            # print("D_demo", D_demo[:10])
            D_s_demo = D_demo[selected_demo]
            D_s_samp = D_samp[selected_samp]

            #D̂ samp ← D̂ demo ∪ D̂ samp
            D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis = 0)

            states, actions, probs,  = D_s_samp[:,:obs_dim], D_s_samp[:,obs_dim:-1], D_s_samp[:,-1]
            # print("states", np.shape(states))
            # print("actions", np.shape(actions))

            states_expert, actions_expert = D_s_demo[:,:obs_dim], D_s_demo[:,obs_dim:-1]

            # Reducing from float64 to float32 for making computaton faster
            states = torch.tensor(states, dtype=torch.float32)
            probs = torch.exp(torch.tensor(probs, dtype=torch.float32))
            actions = torch.tensor(actions, dtype=torch.float32)
            states_expert = torch.tensor(states_expert, dtype=torch.float32)
            actions_expert = torch.tensor(actions_expert, dtype=torch.float32)

            costs_samp = cost_f(torch.cat((states, actions), dim=-1))
            costs_demo = cost_f(torch.cat((states_expert, actions_expert), dim=-1))

            # LOSS CALCULATION FOR IOC (COST FUNCTION)
            loss_IOC = torch.mean(costs_demo) + \
                    torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
            
            # UPDATING THE COST FUNCTION
            cost_optimizer.zero_grad()
            loss_IOC.backward()
            ##mpi_avg_grads(cost_f)    # average grads across MPI processes

            cost_optimizer.step()

            loss_rew.append(loss_IOC.detach())
            
            logger.store(Loss_IOC=loss_IOC)
            # print("loss_IOC: ", loss_IOC.detach())
            # input("cost function updated")
        
        start_ptx = 0
        end_ptx = 0
        ep_cost = 0
        # NOW LET'S CALCULATE THE COST FOR COLLECTED EPISODES
        for idx_s, [obs_s, act_s, done_s] in enumerate(zip(buf.obs_buf, buf.act_buf, buf.done_buf)):
        
            obs_s = torch.tensor(obs_s, dtype=torch.float32)
            act_s = torch.tensor(act_s, dtype=torch.float32)
                
            cost_s = cost_f(torch.cat((obs_s, act_s), dim=-1)).detach().numpy()
            reward_s = -cost_s
            buf.rew_buf[idx_s] = reward_s
            ep_cost += cost_s
            # print("cost_s", cost_s)
            # print("done", done_s)
            # print("idx_s", idx_s)
            if done_s != 0:
                
                end_ptx = idx_s + 1
                print("end_ptx", end_ptx)
                if done_s == 1: # successful done
                    buf.recalculate_adv(last_val=0, start_ptx=start_ptx, end_ptx=end_ptx)
                    print("successful done")
                if done_s == 2:
                    _, v, _ = ac.step(torch.as_tensor(obs_s, dtype=torch.float32))
                    buf.recalculate_adv(last_val=v, start_ptx=start_ptx, end_ptx=end_ptx)
                    print("unsuccessful done")
                start_ptx = end_ptx
                
                logger.store(EpCost=ep_cost)
                ep_cost = 0
            

        # Perform PPO update!
        update()
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Loss_IOC', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='CartPole-v0')
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--env', type=str, default='InmoovGymEnv-v0')
    # parser.add_argument('--env', type=str, default='KukaButtonGymEnv-v0')
    parser.add_argument('--env', type=str, default='JakaLiveGymEnv-v0')
    #parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=24000)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--exp_name', type=str, default='gcl_live_jaka_new')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup2.utils.run_utils import setup_logger_kwargs
    for args.seed in range(0, 3):#0,10
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
        model, env = ppo(lambda : gym.make(args.env, dubug_mode=False, action_repeat=10), actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
            seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
            logger_kwargs=logger_kwargs)
        
        # def generate_session(model, env, t_max=1000):
            
        #     states, traj_probs, actions, rewards = [], [], [], []
        #     s = env.reset()
        #     q_t = 0   # log probability
            
        #     for t in range(t_max):
                
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
