import torch
import argparse, traceback
import time, wandb

import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from torch.optim import Adam
from incremental_rl.logger import Logger
from incremental_rl.td_error_scaler import TDErrorScaler
from gymnasium.wrappers import NormalizeObservation
from incremental_rl.utils import orthogonal_weight_init, set_one_thread
from incremental_rl.experiment_tracker import ExperimentTracker


class Actor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """
    def __init__(self, obs_dim, action_dim, device, n_hid, pnorm):
        super(Actor, self).__init__()
        self.pnorm = pnorm
        self.device = device
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # Two hidden layers
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)


    def forward(self, obs):
        obs = obs.to(self.device)
        fc1 = self.fc1(obs)
        phi = self.fc2(fc1)

        if self.pnorm:
            phi_norm = torch.norm(phi, dim=1).view((-1, 1))
            phi = phi/phi_norm

        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
   
        try:
            dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        except Exception as e:
            print("Mean: {}, Sigma: {}".format(mu, torch.exp(log_std[0])))
            raise e
        
        action_pre = dist.sample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)
        
        # N.B: Tanh must be applied _only_ after lprob estimation of dist sampled action!! 
        #   A mistake here can break learning :/ 
        action = torch.tanh(action_pre)
        action_info = {'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': lprob, 'action_pre': action_pre}

        return action, action_info


class Critic(nn.Module):
    def __init__(self, obs_dim, device, n_hid, pnorm):
        super(Critic, self).__init__()
        self.pnorm = pnorm
        self.device = device

        # Two hidden layers
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),            
        )
        self.value = nn.Linear(n_hid, 1)
        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs):
        fc1 = self.fc1(obs)
        phi = self.fc2(fc1)
        
        if self.pnorm:
            phi_norm = torch.norm(phi, dim=1).view((-1, 1))
            phi = phi/phi_norm
        return self.value(phi).view(-1)
       

class IncrementalActorCritic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = 0  

        self.actor = Actor(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, device=cfg.device, 
                           n_hid=cfg.nhid_actor, pnorm=cfg.pnorm)
        self.critic = Critic(obs_dim=cfg.obs_dim, device=cfg.device, n_hid=cfg.nhid_critic, pnorm=cfg.pnorm)

        self.popt = Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=cfg.betas, weight_decay=cfg.l2_actor)
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr, betas=cfg.betas, weight_decay=cfg.l2_critic)

        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.steps = 0

        self.td_error_scaler = TDErrorScaler()
        self.G = 0

    def compute_action(self, obs):
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, action_info = self.actor(obs)
        return action, action_info

    def update(self, obs, action, next_obs, reward, done, **kwargs):
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs = torch.Tensor(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        obs, action, next_obs = obs.to(self.device), action.to(self.device), next_obs.to(self.device)
        dist, lprob, mu, log_std, action_pre = kwargs['dist'], kwargs['lprob'], kwargs['mu'], kwargs['log_std'], kwargs['action_pre']

        #### Return scaling
        self.G = self.gamma * self.G + reward
        if done:
            self.td_error_scaler.update(reward=reward, gamma=0, G=self.G)
            self.G = 0
        else:
            self.td_error_scaler.update(reward=reward, gamma=self.cfg.gamma, G=None)
        ####

        vs = self.critic(obs)
        vsp = self.critic(next_obs)        

        delta = reward + (1 - done) * self.gamma * vsp.detach() - vs
        if self.cfg.scaled_td:
            delta /= self.td_error_scaler.sigma

        closs = delta ** 2
        pobj = lprob * delta.detach() + (self.alpha * dist.entropy())
        ploss = -pobj

        self.popt.zero_grad()
        ploss.backward()                 
        actor_grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.actor.parameters()]))
        actor_weight_norm = np.sqrt(sum([torch.norm(p.data)**2 for p in self.actor.parameters()]))
        self.popt.step()

        self.critic_opt.zero_grad()
        closs.backward()
        critic_grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in self.critic.parameters()]))
        critic_weight_norm = np.sqrt(sum([torch.norm(p.data)**2 for p in self.critic.parameters()]))
        self.critic_opt.step()

        self.steps += 1
        
        ### Log stats
        stat = {
            'train/actor_loss': ploss.detach().cpu().item(),
            'train/critic_loss': closs.detach().cpu().item(),
            'train/num_updates': self.steps,
            'train/actor_grad_norm': actor_grad_norm.item(),
            'train/critic_grad_norm': critic_grad_norm.item(),
            'train/actor_weight_norm': actor_weight_norm.item(),
            'train/critic_weight_norm': critic_weight_norm.item(),
            'train/mod_action': torch.mean(torch.abs(action)).item(),
            'train/entropy': dist.entropy().item()
        }

        return stat

    def save(self, model_dir, unique_str):
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
        }
        torch.save(
            model, '%s/%s.pt' % (model_dir, unique_str)
        )


def main(args):
    tic = time.time()

    expt = ExperimentTracker(args)
    L = Logger(args.results_dir, prefix=f"{expt.run_id}_", use_tb=False)

    # Env
    env = gym.make(args.env)
    env = gym.wrappers.FlattenObservation(env)
    if args.normalize_obs:
        env = NormalizeObservation(env)

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
    agent = IncrementalActorCritic(args)

    # Weights & Biases; start a new wandb run to track this script
    wandb.init(project="avg", config=vars(args), name=expt.exp_name, entity="gauthamv", 
               mode=args.wandb_mode, dir=f"{args.results_dir}")

    # Interaction     
    rets, ep_steps = [], []
    i_episode, ret, step, ep_entropy, ep_mod_action = 0, 0, 0, 0, 0
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
           
            # Dump training metrics to logger
            stat = agent.update(obs, action, next_obs, reward, terminated, **action_info)            
            for k, v in stat.items():
                L.log(k, v, t)
            ep_entropy += stat['train/entropy']
            ep_mod_action += stat['train/mod_action']

            # Log
            ret += reward
            step += 1

            obs = next_obs

            # Termination
            if terminated or truncated:                
                rets.append(ret)
                ep_steps.append(step)
                i_episode += 1
                ep_entropy /= step; ep_mod_action /= step
                if args.debug:
                    for k, v in stat.items():
                        L.log(k, v, t)                
                    L.log('train/duration', time.time() - ep_tic, t)
                    L.log('train/episode_return', ret, t)
                    L.log('train/episode', len(rets), t)
                    L.print_log(t)
                    stat['entropy'] = ep_entropy; stat["mod_action"] = ep_mod_action
                    expt.dump(t, rets, ep_steps, stat)

                ep_tic = time.time()
                obs, _ = env.reset()
                ret, step, ep_entropy, ep_mod_action = 0, 0, 0, 0                         
    except Exception as e:
        print(e)
        print("Exiting this run, storing partial logs in the database for future debugging...")
        traceback.print_exc()

    if not (terminated or truncated):
        # N.B: We're adding a partial episode just to make plotting easier. But this data point shouldn't be used
        print("Appending partial episode #{}, length: {}, Total Steps: {}".format(i_episode+1, step, t+1))
        rets.append(ret)
        ep_steps.append(step)
        ep_entropy /= step; ep_mod_action /= step
        stat['entropy'] = ep_entropy; stat["mod_action"] = ep_mod_action
    
    # Save returns and args before exiting run
    expt.dump(t, rets, ep_steps, stat)
    if args.save_model:
        agent.save()


    print("Run with id: {} took {:.3f}s!".format(expt.run_id, time.time()-tic))
    wandb.finish()
    return ep_steps, rets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="Hopper-v4", type=str, help="e.g., 'Hopper-v4'")
    parser.add_argument('--seed', default=42, type=int, help="Seed for random number generator")       
    parser.add_argument('--N', default=1001000, type=int, help="# timesteps for the run")
    # SAVG params
    parser.add_argument('--actor_lr', default=0.00006, type=float, help="Actor step size")
    parser.add_argument('--critic_lr', default=0.00087, type=float, help="Critic step size")
    parser.add_argument('--beta1', default=0.9, type=float, help="Beta1 parameter of Adam optimizer")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--alpha_lr', default=0.6, type=float, help="Entropy Coefficient for AVG")
    parser.add_argument('--l2_actor', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--l2_critic', default=0, type=float, help="L2 Regularization")    
    parser.add_argument('--nhid_actor', default=256, type=int)
    parser.add_argument('--nhid_critic', default=256, type=int)
    # Miscellaneous
    parser.add_argument('--checkpoint', default=1000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--results_dir', default="./results", type=str, help="Location to store results")
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--do_not_save', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--wandb_mode', default='disabled', type=str, help="Either online, offline, or disabled")
    parser.add_argument('--debug', action='store_true', default=False)
    # Abaltions args
    parser.add_argument('--normalize_obs', action='store_true', default=False)    
    parser.add_argument('--pnorm', action='store_true', default=False)
    parser.add_argument('--scaled_td', action='store_true', default=False)
    args = parser.parse_args()
    
    # Adam 
    args.betas = [args.beta1, 0.999]

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")    

    if not (args.normalize_obs or args.pnorm or args.scaled_td):
        args.algo = "iac_basic"
    elif args.pnorm and not (args.normalize_obs or args.scaled_td):
        args.algo = "iac_pnorm_only"
    elif args.normalize_obs and not (args.pnorm or args.scaled_td):
        args.algo = "iac_norm_obs_only"
    elif (args.normalize_obs and args.pnorm) and not args.scaled_td:
        args.algo = "iac_norm"
    elif args.normalize_obs and args.pnorm and args.scaled_td:
        args.algo = "iac_norm_scaled"
    else:
        args.algo = "iac_scaled_variant"
    print(args.algo)
    
    # Start experiment
    set_one_thread()
    main(args)