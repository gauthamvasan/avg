import torch, time
import argparse, os, traceback

import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from gymnasium.wrappers import NormalizeObservation, ClipAction
from datetime import datetime
from incremental_rl.experiment_tracker import record_video
from incremental_rl.td_error_scaler import TDErrorScaler


def orthogonal_weight_init(m):
    """ Orthogonal weight initialization for neural networks """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

def human_format_numbers(num, use_float=False):
    # Make human readable short-forms for large numbers
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if use_float:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def set_one_thread():
    '''
    N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
    Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


class Actor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """
    def __init__(self, obs_dim, action_dim, device, n_hid):
        super(Actor, self).__init__()
        self.device = device
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # Two hidden layers
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs):        
        phi = self.phi(obs.to(self.device))
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))       
        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))        
        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)
        
        # N.B: Tanh must be applied _only_ after lprob estimation of dist sampled action!! 
        #   A mistake here can break learning :/ 
        action = torch.tanh(action_pre)
        action_info = {'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': lprob, 'action_pre': action_pre}

        return action, action_info


class Q(nn.Module):
    def __init__(self, obs_dim, action_dim, device, n_hid):
        super(Q, self).__init__()
        self.device = device

        # Two hidden layers
        self.phi = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),            
        )
        self.q = nn.Linear(n_hid, 1)
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1).to(self.device)
        phi = self.phi(x)
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))
        return self.q(phi).view(-1)
       

class AVG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = 0  
        
        self.actor = Actor(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, device=cfg.device, n_hid=cfg.nhid_actor)
        self.Q = Q(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, device=cfg.device, n_hid=cfg.nhid_critic)
        
        self.popt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=cfg.betas)
        self.qopt = torch.optim.Adam(self.Q.parameters(), lr=cfg.critic_lr, betas=cfg.betas)

        self.alpha, self.gamma, self.device = cfg.alpha_lr, cfg.gamma, cfg.device
        self.td_error_scaler = TDErrorScaler()
        self.G = 0

    def compute_action(self, obs):
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, action_info = self.actor(obs)
        return action, action_info

    def update(self, obs, action, next_obs, reward, done, **kwargs):
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs = torch.Tensor(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, lprob = action.to(self.device), kwargs['lprob']

        #### Return scaling
        r_ent = reward - self.alpha * lprob.detach().item()
        self.G += r_ent        
        if done:
            self.td_error_scaler.update(reward=r_ent, gamma=0, G=self.G)
            self.G = 0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.cfg.gamma, G=None)
        ####

        #### Q loss
        q = self.Q(obs, action.detach())    # N.B: Gradient should NOT pass through action here
        with torch.no_grad():
            next_action, action_info = self.actor(next_obs)
            next_lprob = action_info['lprob']
            q2 = self.Q(next_obs, next_action)
            target_V = q2 - self.alpha * next_lprob

        delta = reward + (1 - done) *  self.gamma * target_V - q
        delta /= self.td_error_scaler.sigma
        qloss = delta ** 2
        ####

        # Policy loss
        ploss = self.alpha * lprob - self.Q(obs, action) # N.B: USE reparametrized action
        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        qloss.backward()
        self.qopt.step()

        self.steps += 1


    def save(self, model_dir, unique_str):
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
        }
        torch.save(model, '%s/%s.pt' % (model_dir, unique_str))


def main(args):
    tic = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"-{args.algo}-{args.env}_seed-{args.seed}"

    # Env
    env = gym.make(args.env)
    env = gym.wrappers.FlattenObservation(env)
    env = NormalizeObservation(env)
    env = ClipAction(env)

    #### Reproducibility
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    # Learner
    args.obs_dim =  env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    agent = AVG(args)

    # Interaction     
    rets, ep_steps = [], []
    ret, step = 0, 0
    terminated, truncated = False, False
    obs, _ = env.reset()
    ep_tic = time.time()    
    try:
        for t in range(args.N):
            # N.B: Action is a torch.Tensor
            action, action_info = agent.compute_action(obs)                
            sim_action = action.detach().cpu().view(-1).numpy()

            # Receive reward and next state
            next_obs, reward, terminated, truncated, _ = env.step(sim_action)
            agent.update(obs, action, next_obs, reward, terminated, **action_info)            
            ret += reward
            step += 1

            obs = next_obs

            if t % args.checkpoint == 0 and args.save_model:
                agent.save(model_dir=args.results_dir, unique_str=f"{run_id}_model_{human_format_numbers(t)}")

            # Termination
            if terminated or truncated:
                rets.append(ret)
                ep_steps.append(step)
                print("E: {}| D: {:.3f}| S: {}| R: {:.2f}| T: {}".format(len(rets), time.time() - ep_tic, step, ret, t))

                ep_tic = time.time()
                obs, _ = env.reset()
                ret, step = 0, 0
    except Exception as e:
        print(e)
        print("Exiting this run, storing partial logs in the database for future debugging...")
        traceback.print_exc()

    if not (terminated or truncated):
        # N.B: We're adding a partial episode just to make plotting easier. But this data point shouldn't be used
        print("Appending partial episode #{}, length: {}, Total Steps: {}".format(len(rets), step, t+1))
        rets.append(ret)
        ep_steps.append(step)
    
    # Save returns and args before exiting run
    if args.save_model:
        agent.save(model_dir=args.results_dir, unique_str=f"{run_id}_model")


    print("Run with id: {} took {:.3f}s!".format(run_id, time.time()-tic))
    
    # Eval
    if args.n_eval:
        record_video(env, agent, num_episodes=args.n_eval, video_filename=f"{args.results_dir}/{run_id}.avi")

    return ep_steps, rets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="Humanoid-v4", type=str, help="e.g., 'Humanoid-v4'")
    parser.add_argument('--seed', default=42, type=int, help="Seed for random number generator")       
    parser.add_argument('--N', default=10001000, type=int, help="# timesteps for the run")
    # SAVG params
    parser.add_argument('--actor_lr', default=0.0063, type=float, help="Actor step size")
    parser.add_argument('--critic_lr', default=0.0087, type=float, help="Critic step size")
    parser.add_argument('--beta1', default=0., type=float, help="Beta1 parameter of Adam optimizer")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--alpha_lr', default=0.07, type=float, help="Entropy Coefficient for AVG")
    parser.add_argument('--l2_actor', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--l2_critic', default=0, type=float, help="L2 Regularization")    
    parser.add_argument('--nhid_actor', default=256, type=int)
    parser.add_argument('--nhid_critic', default=256, type=int)
    # Miscellaneous
    parser.add_argument('--checkpoint', default=50000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--results_dir', default="./results", type=str, help="Location to store results")
    parser.add_argument('--device', default="cpu", type=str)    
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--n_eval', default=0, type=int, help="Number of eval episodes")
    args = parser.parse_args()
    
    # Adam 
    args.betas = [args.beta1, 0.999]

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")    

    args.algo = "AVG"
    
    # Start experiment
    set_one_thread()
    main(args)