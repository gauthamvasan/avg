import torch
import argparse, traceback
import time, wandb

import numpy as np
import gymnasium as gym

from copy import deepcopy
from torch.optim import Adam
from incremental_rl.logger import Logger
from incremental_rl.experiment_tracker import ExperimentTracker
from incremental_rl.SAC.cnn_policies import ImageEncoder, Actor, DoubleQ
from incremental_rl.SAC.rad_buffer import SACRADBuffer, RandomShiftsAug
from incremental_rl.SAC.min_time_dm_control import ReacherWrapper, BallInCupWrapper


class SAC_RAD:
    """ SAC algorithm. """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau        

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr

        self.action_dim = cfg.action_dim
        
        encoder = ImageEncoder(image_shape=cfg.image_shape, device=cfg.device)
        self.actor = Actor(encoder=encoder, proprioception_dim=cfg.proprioception_dim, action_dim=cfg.action_dim, n_hid=cfg.nhid_actor, device=cfg.device)
        self.critic = DoubleQ(encoder=encoder, proprioception_dim=cfg.proprioception_dim, action_dim=cfg.action_dim, n_hid=cfg.nhid_critic, device=cfg.device)
        self.critic_target = deepcopy(self.critic) # also copies the encoder instance

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(cfg.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -cfg.action_dim

        self.num_updates = 0

        # optimizers
        self.init_optimizers()
        self.train()
        self.critic_target.train()
               
        self.steps = 0
        self.rad = RandomShiftsAug(pad=4)

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
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.cfg.betas)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.cfg.betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def compute_action(self, image, proprioception, detach_encoder=True):
        # Convert the RGB image to a PyTorch tensor
        image = torch.Tensor(image.astype(np.float32)).unsqueeze(0).to(self.device)
        proprioception = torch.Tensor(proprioception.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, info = self.actor(image, proprioception, detach_encoder)
        return  action.cpu().data.numpy().flatten(), info

    def update_critic(self, img, proprioception, action, next_img, next_proprioception, reward, done):
        with torch.no_grad():
            policy_action, info = self.actor(next_img, next_proprioception)
            target_Q1, target_Q2 = self.critic_target(next_img, next_proprioception, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * info['lprob']
            if self.cfg.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = reward + (self.cfg.gamma * target_V)
            else:
                target_Q = reward + ((1.0 - done) * self.cfg.gamma * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(img, proprioception, action)
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        critic_stats = {
            'train/critic_loss': critic_loss.item()
        }

        return critic_stats

    def update_actor_and_alpha(self, img, proprioception):
        # detach encoder, so we don't update it with the actor loss
        action, info = self.actor(img, proprioception, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(img, proprioception, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * info['lprob'] - actor_Q).mean()

        entropy = 0.5 * info['log_std'].shape[1] * (1.0 + np.log(2 * np.pi)) + info['log_std'].sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-info['lprob'] - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train/actor_loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy,
            'train_actor/entropy': entropy.mean().item(),
            'train/ent_loss': alpha_loss.item(),
            'train/ent_alpha': self.alpha.item(),
            'train/entropy': entropy.mean().item(),
            'train/mod_action': torch.mean(torch.abs(action)).item(),
        }
        return actor_stats

    def update(self, images, proprioception, action, next_images, next_proprioception, reward, done):
        # Move tensors to device
        images = self.rad(images); next_images = self.rad(next_images) 
        images, proprioception = images.to(self.device), proprioception.to(self.device)
        next_images, next_proprioception = next_images.to(self.device), next_proprioception.to(self.device)
        actions, rewards, dones = actions.to(self.device), rewards.to(self.device), dones.to(self.device)
            

        # regular update of SAC_RAD, sequentially augment data and train
        stats = self.update_critic(images, proprioception, action, next_images, next_proprioception, reward, done)        
        
        actor_stats = self.update_actor_and_alpha(images, proprioception)        
        stats = {**stats, **actor_stats}        

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
        self.soft_update_params(
            self.critic.encoder, self.critic_target.encoder,
            self.encoder_tau
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



class SACRADAgent(SAC_RAD):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Replay buffer
        self._replay_buffer = SACRADBuffer(cfg.image_shape, cfg.proprioception_shape, cfg.action_shape, cfg.replay_buffer_capacity, cfg.batch_size)
        self.steps = 0

    def update(self, images, proprioception, action, next_images, next_proprioception, reward, done, **kwargs):
        self._replay_buffer.add(images, proprioception, action, reward, done)        
        self.steps += 1
        
        stats = {}
        if self.steps > self.cfg.init_steps:
            for _ in range(self.cfg.update_epochs):
                tic = time.time()
                images, proprioceptions, actions, next_images, next_proprioceptions, rewards, dones = self._replay_buffer.sample()
                images = self.rad(images); next_images = self.rad(next_images) 
                images, proprioceptions = images.to(self.device), proprioceptions.to(self.device)
                next_images, next_proprioceptions = next_images.to(self.device), next_proprioceptions.to(self.device)
                actions, rewards, dones = actions.to(self.device), rewards.to(self.device), dones.to(self.device)

                # regular update of SAC_RAD, sequentially augment data and train
                stats = self.update_critic(images, proprioceptions, actions, next_images, next_proprioceptions, rewards, dones)
                
                if self.steps % self.cfg.update_actor_every == 0:
                    actor_stats = self.update_actor_and_alpha(images, proprioceptions)        
                    stats = {**stats, **actor_stats}        

                if self.steps % self.cfg.update_critic_target_every == 0:
                    self.soft_update_target()
                
                stats['train/batch_reward'] = rewards.mean().item()
                stats['train/num_updates'] = self.num_updates
                self.num_updates += 1

                if self.num_updates %100 == 0:
                    print("Update {} took {:.3f}s".format(self.num_updates, time.time() - tic))
        return stats


def main(args):
    tic = time.time()

    expt = ExperimentTracker(args)
    L = Logger(args.results_dir, prefix=f"{expt.run_id}_", use_tb=False)

    # Env
    env = ReacherWrapper(seed=args.seed, timeout=100, reward=-1, mode="hard", use_image=True, img_history=3)

    #### Reproducibility
    # env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    # Learner
    args.proprioception_dim =  env.proprioception_space.shape[0]
    args.proprioception_shape =  env.proprioception_space.shape
    args.action_dim = env.action_space.shape[0]
    args.action_shape = env.action_space.shape
    args.image_shape = env.image_space.shape
    agent = SACRADAgent(args)

    # Weights & Biases; start a new wandb run to track this script
    wandb.init(project="avg", config=vars(args), name=expt.exp_name, entity="gauthamv", 
               mode=args.wandb_mode, dir=f"{args.results_dir}")

    # Interaction     
    rets, ep_steps = [], []
    i_episode, ret, step, ep_mod_action = 0, 0, 0, 0
    terminated, truncated = False, False
    obs, _ = env.reset()    
    ep_tic = time.time()    
    try:
        for t in range(args.N):
            # N.B: Action is a torch.Tensor          
            action, action_info = agent.compute_action(obs.images, obs.proprioception)            

            # Receive reward and next state
            next_obs, reward, terminated, truncated, _ = env.step(action)            
           
            # Dump training metrics to logger
            stat = agent.update(obs.images, obs.proprioception, action, next_obs.images, next_obs.proprioception, reward, terminated, **action_info)            
            for k, v in stat.items():
                L.log(k, v, t)            
            ep_mod_action += np.mean(np.abs(action))

            # Log
            ret += reward
            step += 1

            obs = next_obs

            # Termination
            if terminated or truncated:
                if truncated:
                    obs, _ = env.reset(randomize_target=False)
                    continue

                L.log('train/duration', time.time() - ep_tic, t)
                L.log('train/episode_return', ret, t)
                L.log('train/episode', len(rets), t)
                L.print_log(t)

                rets.append(ret)
                ep_steps.append(step)
                i_episode += 1
                ep_mod_action /= step

                # Log to file if necessary
                stat["mod_action"] = ep_mod_action
                expt.dump(t, rets, ep_steps, stat)

                ep_tic = time.time()
                obs, _ = env.reset()
                ret, step, ep_mod_action = 0, 0, 0                         
    except Exception as e:
        print(e)
        print("Exiting this run, storing partial logs in the database for future debugging...")
        traceback.print_exc()

    if not (terminated or truncated):
        # N.B: We're adding a partial episode just to make plotting easier. But this data point shouldn't be used
        print("Appending partial episode #{}, length: {}, Total Steps: {}".format(i_episode+1, step, t+1))
        rets.append(ret)
        ep_steps.append(step)
        ep_mod_action /= step
        stat["mod_action"] = ep_mod_action
    
    # Save returns and args before exiting run
    expt.dump(t, rets, ep_steps, stat)
    if args.save_model:
        agent.save()


    print("Run with id: {} took {:.3f}s!".format(expt.run_id, time.time()-tic))
    wandb.finish()
    return ep_steps, rets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="Reacher-v4", type=str, help="e.g., 'Hopper-v4'")
    parser.add_argument('--seed', default=42, type=int, help="Seed for random number generator")       
    parser.add_argument('--N', default=1001000, type=int, help="# timesteps for the run")
    # SAC params
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="Actor step size")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="Critic step size")
    parser.add_argument('--beta1', default=0.9, type=float, help="Beta1 parameter of Adam optimizer")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--alpha_lr', default=0.0003, type=float, help="Entropy Coefficient for AVG")
    parser.add_argument('--l2_actor', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--l2_critic', default=0, type=float, help="L2 Regularization")    
    parser.add_argument('--nhid_actor', default=256, type=int)
    parser.add_argument('--nhid_critic', default=256, type=int)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--encoder_tau', default=0.01, type=float)
    parser.add_argument('--init_steps', default=1000, type=int)        
    parser.add_argument('--update_epochs', default=1, type=int)        
    parser.add_argument('--update_actor_every', default=2, type=int)        
    parser.add_argument('--update_critic_target_every', default=1, type=int)
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)    
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    # Miscellaneous
    parser.add_argument('--checkpoint', default=10000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--results_dir', default="./results", type=str, help="Location to store results")
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--do_not_save', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--description', default='', type=str)
    parser.add_argument('--wandb_mode', default='disabled', type=str, help="Either online, offline, or disabled")
    args = parser.parse_args()
    
    # Adam 
    args.betas = [args.beta1, 0.999]

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")    

    args.algo = "sac_rad"
    
    # Start experiment
    main(args)