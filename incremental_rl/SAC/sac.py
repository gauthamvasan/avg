import torch
import time, argparse
import wandb, traceback

import numpy as np
import gymnasium as gym

from copy import deepcopy
from incremental_rl.logger import Logger
from incremental_rl.SAC.mlp_policies import SquashedGaussianMLPActor, DoubleQ
from incremental_rl.SAC.rad_buffer import SACReplayBuffer
from incremental_rl.experiment_tracker import ExperimentTracker
from incremental_rl.utils import set_one_thread


class SAC:
    """ SAC with Automatic Entropy Adjustment. """
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.update_critic_target_every = cfg.update_critic_target_every

        self.actor = SquashedGaussianMLPActor(cfg.obs_dim, cfg.action_dim, cfg.actor_nn_params, device)
        self.critic = DoubleQ(cfg.obs_dim, cfg.action_dim, cfg.critic_nn_params, device)
        self.critic_target = deepcopy(self.critic) # also copies the encoder instance

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -cfg.action_dim

        self.num_updates = 0

        # optimizers
        self.init_optimizers()
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        self.log_alpha.share_memory_()

    def init_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor_lr, betas=self.cfg.betas,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic_lr, betas=self.cfg.betas,
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.cfg.alpha_lr, betas=(0.5, 0.999),
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_critic(self, obs, action, next_obs, reward, done):
        with torch.no_grad():
            next_action, action_info = self.actor(next_obs, rp=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * action_info['lprob']            
            if self.cfg.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = reward + (self.gamma * target_V)
            else:
                target_Q = reward + ((1.0 - done) * self.gamma * target_V)
        
        # get current Q estimates       
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = np.sqrt(sum([(torch.norm(p.grad.cpu())**2) for p in self.critic.parameters()]))
        critic_weight_norm = np.sqrt(sum([(torch.norm(p.data.cpu())**2) for p in self.critic.parameters()]))
        self.critic_optimizer.step()

        critic_stats = {
            'train/critic_loss': critic_loss.item(),
            'train/critic_grad_norm': critic_grad_norm.item(),
            'train/critic_weight_norm': critic_weight_norm.item(),
        }

        return critic_stats

    def update_actor_and_alpha(self, obs):        
        # detach encoder, so we don't update it with the actor loss
        action, action_info = self.actor(obs, rp=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * action_info['lprob'] - actor_Q).mean()

        entropy = 0.5 * action_info['log_std'].shape[1] * (1.0 + np.log(2 * np.pi)) + action_info['log_std'].sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()        
        actor_loss.backward()
        actor_grad_norm = np.sqrt(sum([(torch.norm(p.grad.cpu())**2) for p in self.actor.parameters()]))
        actor_weight_norm = np.sqrt(sum([torch.norm(p.data.cpu())**2 for p in self.actor.parameters()]))
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-action_info['lprob'] - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        alpha_grad_norm = np.sqrt((torch.norm(self.log_alpha.grad)**2).to('cpu'))        
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train/actor_loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy,
            'train/ent_loss': alpha_loss.item(),
            'train/ent_alpha': self.alpha.item(),
            'train/entropy': entropy.mean().item(),
            'train/actor_grad_norm': actor_grad_norm.item(),
            'train/actor_weight_norm': actor_weight_norm.item(),
            'train/alpha_grad_norm': alpha_grad_norm.item(),
        }
        return actor_stats

    def update(self, obs, action, next_obs, reward, done):
        # Move tensors to device
        obs, action, next_obs, reward, done = obs.to(self.device), action.to(self.device), \
            next_obs.to(self.device), reward.to(self.device), done.to(self.device)

        # Update critic
        stats = self.update_critic(obs, action, next_obs, reward, done)

        # Update actor and alpha
        actor_stats = self.update_actor_and_alpha(obs)        
        stats = {**stats, **actor_stats}

        if self.num_updates % self.update_critic_target_every == 0:
            self.soft_update_target()
        
        stats['train/batch_reward'] = reward.mean().item()
        stats['train/num_updates'] = self.num_updates
        self.num_updates += 1
        return stats

    @staticmethod
    def soft_update_params(net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def soft_update_target(self):
        self.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        self.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
    
    def save(self, model_dir, unique_str):
        model_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().item(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "log_alpha_opt": self.log_alpha_optimizer.state_dict(),
        }
        torch.save(
            model_dict, '%s/%s.pt' % (model_dir, unique_str)
        )

    def load(self, model_dir, unique_str):
        model_dict = torch.load('%s/%s.pt' % (model_dir, unique_str))
        self.actor.load_state_dict(model_dict["actor"])
        self.critic.load_state_dict(model_dict["critic"])
        self.log_alpha = torch.tensor(model_dict["log_alpha"]).to(self.device)
        self.log_alpha.requires_grad = True
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
        self.log_alpha_optimizer.load_state_dict(model_dict["log_alpha_optimizer"])


class SACAgent(SAC):
    def __init__(self, cfg):
        super().__init__(cfg, cfg.device)
        # Replay buffer
        self._replay_buffer = SACReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.replay_buffer_capacity, cfg.batch_size)
        self.steps = 0

    def update(self, obs, action, next_obs, reward, done, **kwargs):
        self._replay_buffer.add(obs, action, next_obs, reward, done)        
        self.steps += 1
        
        stat = {}
        if self.steps > self.cfg.init_steps and (self.steps % self.cfg.update_every == 0):
            for _ in range(self.cfg.update_epochs):
                tic = time.time()
                stat = super().update(*self._replay_buffer.sample())
                # if self.num_updates %100 == 0:
                    # print(f"Update {self.num_updates} took {time.time() - tic}s")
        return stat

    def compute_action(self, obs):
        with torch.no_grad():
            if not isinstance(obs, torch.FloatTensor):
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
            action, action_info = self.actor(obs, rp=True)
            return action.cpu(), action_info  


def main(args):
    tic = time.time()

    expt = ExperimentTracker(args)
    L = Logger(args.results_dir, prefix=f"{expt.run_id}_", use_tb=False)

    # Env
    env = gym.make(args.env)

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
    agent = SACAgent(args)

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
            if args.debug:
                if stat is not None:
                    for k, v in stat.items():
                        L.log(k, v, t)

            ep_entropy += action_info['entropy']
            ep_mod_action += action_info['mod_action']

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
                    L.log('train/duration', time.time() - ep_tic, t)
                    L.log('train/episode_return', ret, t)
                    L.log('train/episode', len(rets), t)
                    L.print_log(t)

                    # Log to file if necessary
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
    if args.debug:
        expt.dump(t, rets, ep_steps, stat)
        
    if args.save_model:
        agent.save()

    print("Run with id: {} took {:.3f}s!".format(expt.run_id, time.time()-tic))
    wandb.finish()
    return ep_steps, rets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="Humanoid-v4", type=str, help="e.g., 'Hopper-v4'")
    parser.add_argument('--seed', default=42, type=int, help="Seed for random number generator")       
    parser.add_argument('--N', default=10001000, type=int, help="# timesteps for the run")
    # SAC params
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="Actor step size")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="Critic step size")
    parser.add_argument('--alpha_lr', default=0.0003, type=float, help="Entropy coefficient step size")
    parser.add_argument('--beta1', default=0.9, type=float, help="Beta1 parameter of Adam optimizer")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")        
    parser.add_argument('--l2_actor', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--l2_critic', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--init_steps', default=100, type=int)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--update_critic_target_every', default=1, type=int)
    parser.add_argument('--update_epochs', default=1, type=int)
    # MLP params
    parser.add_argument('--actor_hidden_sizes', default="256,256", type=str)
    parser.add_argument('--critic_hidden_sizes', default="256,256", type=str)
    parser.add_argument('--nn_activation', default="relu", type=str, help="Activations for neural network")
    # Miscellaneous
    parser.add_argument('--checkpoint', default=10000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--results_dir', default="./results", type=str, help="Location to store results")
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--do_not_save', action='store_true', default=False)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--wandb_mode', default='disabled', type=str, help="Either online, offline, or disabled")
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    
    # Adam 
    args.betas = [args.beta1, 0.999]

    args.actor_nn_params = {
        'mlp': {
            'hidden_sizes': list(map(int, args.actor_hidden_sizes.split(','))),
            'activation': args.nn_activation,
        }
    }
    args.critic_nn_params = {
        'mlp': {
            'hidden_sizes': list(map(int, args.critic_hidden_sizes.split(','))),
            'activation': args.nn_activation,
        }
    }

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")    

    args.algo = "sac"
    if args.replay_buffer_capacity == 1 and args.batch_size == 1:
        args.algo = "isac"

    # Start experiment
    set_one_thread()
    main(args)
